import os

# 原始数据路径（Linux）
old_base = "/home/a631/hzz/WeNet/aishell_data//data_aishell/wav"

# Windows 数据路径
new_base = "D:/claude/data_aishell/data_aishell/wav"

# 修复 wav.scp 文件
for split in ['train', 'dev', 'test']:
    scp_file = f'D:/claude/wenet/examples/aishell/s0/data/{split}/wav.scp'
    data_list = f'D:/claude/wenet/examples/aishell/s0/data/{split}/data.list'

    # 备份原始文件
    if os.path.exists(scp_file):
        with open(scp_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换路径
        new_content = content.replace(old_base, new_base)

        with open(scp_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed {scp_file}")

    # 修复 data.list 文件
    if os.path.exists(data_list):
        with open(data_list, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换路径
        new_content = content.replace(old_base, new_base)

        with open(data_list, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed {data_list}")

print("Done!")
