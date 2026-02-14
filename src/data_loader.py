import torch
import numpy as np
from sklearn.datasets import load_digits
from scipy.ndimage import rotate, shift
from sklearn.model_selection import train_test_split

def prepare_and_save_data():
    digits = load_digits()
    mask = np.isin(digits.target, [0, 1])
    X = digits.data[mask]
    y = digits.target[mask] # 0 or 1
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练集：0° + 随机平移噪声
    train_data = []
    for x in X_train_raw:
        img = x.reshape(8, 8)
        # 模拟 0° 时的平移干扰
        img = shift(img, [np.random.randint(-1, 2), np.random.randint(-1, 2)], mode='constant', cval=0)
        train_data.append(img)
    
    # 测试集：全角度旋转
    test_dict = {}
    angles = range(0, 361, 30)
    for ang in angles:
        rot_images = [rotate(x.reshape(8,8), ang, reshape=False, mode='constant', cval=0) for x in X_test_raw]
        test_dict[ang] = (np.array(rot_images), y_test)

    # 保存为 PyTorch 格式
    data_pack = {
        'train_x': np.array(train_data),
        'train_y': y_train,
        'test_dict': test_dict
    }
    torch.save(data_pack, 'shared_digits_data.pt')
    print("数据包 'shared_digits_data.pt' 已生成。包含 0° 训练集和全角度测试集。")

if __name__ == "__main__":
    prepare_and_save_data()