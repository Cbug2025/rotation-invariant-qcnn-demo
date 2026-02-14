import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ================= 1. 黄金配置 (Best Configuration) =================
n_qubits = 16
dev = qml.device("lightning.qubit", wires=n_qubits)

# 经过调试的最佳超参数
CONFIG = {
    "n_layers": 4,        # 4层足够捕捉特征，又不会过拟合
    "batch_size": 8,      # 每次8个样本，梯度估计较准
    "steps": 80,          # 80步，确保 Loss 彻底收敛
    "lr": 0.04,           # 学习率
    "save_path": "q_model_weights.pt" # 保存路径
}

# ================= 2. 等变电路 (The Core) =================
@qml.qnode(dev, interface="torch")
def equivariant_circuit(inputs, weights):
    # Encoding (4x4)
    for i in range(16): qml.RX(inputs[i], wires=i)
    
    # Variational Layers
    for l in range(weights.shape[0]):
        # [拓扑对称性]
        # 内圈旋转 (共享参数 0)
        for i in [5, 6, 9, 10]: qml.RY(weights[l, 0], wires=i)
        # 外圈旋转 (共享参数 1)
        for i in [0,1,2,3,7,11,15,14,13,12,8,4]: qml.RY(weights[l, 1], wires=i)
        
        # [对称纠缠]
        # 内圈环 (共享参数 2)
        qml.CRZ(weights[l, 2], wires=[5, 6]); qml.CRZ(weights[l, 2], wires=[6, 10])
        qml.CRZ(weights[l, 2], wires=[10, 9]); qml.CRZ(weights[l, 2], wires=[9, 5])
        
        # 外圈环 (共享参数 3)
        outer = [0,1,2,3,7,11,15,14,13,12,8,4]
        for k in range(len(outer)): 
            qml.CRZ(weights[l, 3], wires=[outer[k], outer[(k+1)%len(outer)]])
        
        # 内外连接 (共享参数 4)
        qml.CRZ(weights[l, 4], wires=[0, 5]); qml.CRZ(weights[l, 4], wires=[3, 6])
        qml.CRZ(weights[l, 4], wires=[12, 9]); qml.CRZ(weights[l, 4], wires=[15, 10])
    
    # [不变测量] 总磁矩
    obs = qml.PauliZ(0)
    for i in range(1, 16): obs = obs + qml.PauliZ(i)
    return qml.expval(obs)

# ================= 3. 主程序 =================
def run_best_model():
    # --- A. 数据检查 ---
    if not os.path.exists('shared_digits_data.pt'):
        print("❌ 错误：未找到数据文件 'shared_digits_data.pt'。")
        print("请先运行 data_loader.py 生成数据！")
        return

    print(f"📥 正在加载数据...")
    data = torch.load('shared_digits_data.pt', weights_only=False)
    tx, ty = data['train_x'], data['train_y']
    
    # 预处理函数
    def q_prep(imgs):
        # 8x8 -> 4x4 -> Normalize to [0, pi]
        imgs_4x4 = imgs.reshape(-1, 4, 2, 4, 2).mean(4).mean(2).reshape(-1, 16)
        return torch.tensor(imgs_4x4 / 16.0 * np.pi, dtype=torch.float32)

    # --- B. 模型初始化 ---
    weights = torch.randn(CONFIG["n_layers"], 5, requires_grad=True)
    opt = torch.optim.Adam([weights], lr=CONFIG["lr"])
    
    print(f"\n🚀 开始训练 (Target: >90% Accuracy)")
    print(f"配置: Layers={CONFIG['n_layers']} | Steps={CONFIG['steps']} | Batch={CONFIG['batch_size']}")
    
    start_time = time.time()
    loss_history = []

    # --- C. 训练循环 ---
    for step in range(CONFIG["steps"]):
        opt.zero_grad()
        
        # Mini-batch
        batch_idx = np.random.choice(len(tx), CONFIG["batch_size"])
        x_batch = tx[batch_idx]
        y_batch = ty[batch_idx]
        
        # 目标: 0 -> +1.0, 1 -> -1.0
        target = torch.tensor(np.where(y_batch == 0, 1.0, -1.0), dtype=torch.float32)
        
        # Forward
        x_ready = q_prep(x_batch)
        preds = torch.stack([equivariant_circuit(x, weights) for x in x_ready]) / 16.0
        
        loss = torch.mean((preds - target)**2)
        loss.backward()
        opt.step()
        
        loss_history.append(loss.item())
        
        if step % 10 == 0 or step == CONFIG["steps"]-1:
            elapsed = time.time() - start_time
            print(f"Step {step:02d}/{CONFIG['steps']} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

    # --- D. 保存模型 ---
    torch.save(weights, CONFIG["save_path"])
    print(f"\n💾 模型参数已保存至: {CONFIG['save_path']}")

    # --- E. 终极泛化测试 ---
    print("\n⚔️ 开始全角度泛化测试 (Zero-Shot)...")
    angles, accs = [], []
    sorted_angles = sorted(data['test_dict'].keys())
    
    for ang in sorted_angles:
        test_x, test_y = data['test_dict'][ang]
        with torch.no_grad():
            preds = torch.stack([equivariant_circuit(x, weights) for x in q_prep(test_x)])
            # Pred > 0 -> Class 0; Pred < 0 -> Class 1
            correct = ((preds > 0) == (torch.tensor(test_y) == 0)).float().mean()
            acc = correct.item() * 100
            accs.append(acc)
            angles.append(ang)
            
            # 状态指示
            status = "🔥 Perfect" if acc > 95 else ("✅ Good" if acc > 80 else "⚠️ Weak")
            print(f"Angle {ang:3d}° | Accuracy: {acc:.1f}%  {status}")
    
    # --- F. 绘图 ---
    avg_acc = np.mean(accs)
    plt.figure(figsize=(10, 6))
    plt.plot(angles, accs, 'o-', color='#D32F2F', linewidth=2, label=f'Equivariant QML (Avg: {avg_acc:.1f}%)')
    plt.axhline(y=100, color='green', linestyle=':', alpha=0.3)
    plt.axhline(y=90, color='orange', linestyle='--', alpha=0.3, label='Target Threshold')
    plt.ylim(0, 105)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Rotation Angle (Degrees)')
    plt.title(f"Final Model Performance\nTrained on 0°, Tested on All Angles")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    save_fig_name = 'best_model_result.png'
    plt.savefig(save_fig_name)
    print(f"\n📈 结果图表已保存为 '{save_fig_name}'。去看看吧！")
    plt.show()

if __name__ == "__main__":
    run_best_model()