# src/training.py
import math, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from curriculum import lowpass_inputs
from regularizers import total_variation_phase, amplitude_l2

def train_model(name, model, train_loader, val_loader, device,
                epochs=20, base_lr=3e-3, weight_decay=5e-4,
                use_amp=False, curriculum=False,
                tv_max=1e-4, amp_l2_max=1e-4, cosine_per_batch=True):
    
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    
    if cosine_per_batch:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=base_lr*1e-2)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=base_lr*1e-2)

    scaler = GradScaler('cuda', enabled=(use_amp and torch.cuda.is_available()))
    
    # --- HISTORY LOGGING ---
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best = 0.0
    for ep in range(epochs):
        model.train()
        start = time.perf_counter()
        total=correct=0; loss_sum=0.0

        cutoff = 0.25 + 0.75 * (ep / max(1, epochs-1))**1.5
        w_tv  = tv_max   * (1.0 - ep / max(1, epochs-1))
        w_amp = amp_l2_max * (1.0 - ep / max(1, epochs-1))

        for x,y in train_loader:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if curriculum: x = lowpass_inputs(x, cutoff)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=(use_amp and torch.cuda.is_available())):
                logits = model(x)
                ce = crit(logits, y)
                
                # Regularizers
                tv = total_variation_phase(getattr(model, 'phase_masks', [])) if hasattr(model, 'phase_masks') else torch.tensor(0.0, device=x.device)
                loss = ce + w_tv*tv

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            if cosine_per_batch: sched.step()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            total += bs
            correct += (logits.argmax(1) == y).sum().item()

        if not cosine_per_batch: sched.step()

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)
        train_acc = correct/total
        train_loss = loss_sum/total
        
        # Store history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{name}] ep {ep:02d} | tr_acc {train_acc:.3f} | val_acc {val_acc:.3f} | lr {sched.get_last_lr()[0]:.2e}")
        best = max(best, val_acc)

    # Inference Speed Test
    model.eval()
    n_imgs, inf_start = 0, time.perf_counter()
    with torch.no_grad():
        for x,_ in val_loader:
            x = x.to(device)
            _ = model(x)
            n_imgs += x.size(0)
    imgs_per_s = n_imgs / (time.perf_counter() - inf_start)

    return {
        "best_val_acc": best,
        "inference_imgs_per_s": imgs_per_s,
        "history": history # Return the full logs
    }

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total
