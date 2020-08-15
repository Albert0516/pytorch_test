import glob
from tqdm import tqdm
import time
from os import listdir
from os.path import splitext


# glob 测试
for name in glob.glob('./*.py'):
    print(name)

# 列表解析
ids = [splitext(file) for file in listdir('E://BaiduNetdiskDownload//Face-Occlusion-Detect//data//cofw')]
print(ids)

# 进度条
for i in tqdm(range(200), desc="ygy", ncols=100):
    time.sleep(0.05)
    pass
