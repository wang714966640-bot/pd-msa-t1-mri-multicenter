import os
import pandas as pd

# 定义要提取的结构列表
STRUCTURES = [
    "head_of_caudate",
    "body_of_caudate",
    "tail_of_caudate",
    "external_segment_of_globus_pallidus",
    "ventral_pallidus",
    "internal_segment_of_globus_pallidus",
    "putamen",
    "claustrum",
    "basal_nucleus_of_meynert",
    "thalamus",
    "midbrain_(mesencephalon)",
    "pons",
    "medulla_oblongata",
    "red_nucleus",
    "substantia_nigra__reticular_part",
    "substantia_nigra__compact_part",
    "ventral_tegmental_area",
    "pontine_nucleus",
    "inferior_olive__principal_nucleus",
    "superior_cerebellar_peduncle_(brachium_conjunctivum)",
    "inferior_cerebellar_peduncle"
]

def extract_histo_volumes(file_path, side):
    """从vols.left.csv或vols.right.csv中提取特定结构的体积"""
    histo_data = {}
    try:
        df = pd.read_csv(file_path, header=None)
        if len(df) >= 2:
            structures = df.iloc[0].values
            volumes = df.iloc[1].values
            structure_volume_map = dict(zip(structures, volumes))

            for structure in STRUCTURES:
                for key in structure_volume_map:
                    if structure.lower() in key.lower():
                        histo_data[f"{side}_{structure}"] = structure_volume_map[key]
                        break
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return histo_data

def process_subject(subject_dir):
    """处理单个被试的数据，为缺失值填充0"""
    subject_data = {}

    # 初始化所有结构（左右）为0
    for structure in STRUCTURES:
        subject_data[f"left_{structure}"] = 0.0
        subject_data[f"right_{structure}"] = 0.0

    # 提取实际数据（覆盖0值）
    left_file = os.path.join(subject_dir, 'output_histo', 'vols.left.csv')
    if os.path.exists(left_file):
        left_data = extract_histo_volumes(left_file, 'left')
        subject_data.update(left_data)

    right_file = os.path.join(subject_dir, 'output_histo', 'vols.right.csv')
    if os.path.exists(right_file):
        right_data = extract_histo_volumes(right_file, 'right')
        subject_data.update(right_data)

    return subject_data

def main():
    # 读取被试列表（保持原始顺序）
    sublist_path = '/media/neurox/T7_Shield/PD/download_yanzheng_xuanwu/demo_xuanwuyanzheng_1.txt'
    with open(sublist_path, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    # 按原始顺序处理所有被试
    all_data = []
    for subject in subjects:
        subject_dir = os.path.join('/media/neurox/T7_Shield/PD/download_yanzheng_xuanwu', subject)
        if os.path.isdir(subject_dir):
            subject_data = process_subject(subject_dir)
            subject_data['subject_id'] = subject
            all_data.append(subject_data)
        else:
            print(f"Warning: Subject directory not found: {subject_dir}")
            # 为缺失的被试创建全0记录
            empty_data = {'subject_id': subject}
            for structure in STRUCTURES:
                empty_data[f"left_{structure}"] = 0.0
                empty_data[f"right_{structure}"] = 0.0
            all_data.append(empty_data)

    # 转换为DataFrame并严格按原始顺序排列
    df = pd.DataFrame(all_data)
    df.set_index('subject_id', inplace=True)

    # 确保顺序与原始列表一致
    df = df.reindex(subjects)

    # 保存结果（不包含额外的索引列）
    output_path = os.path.join('/media/neurox/T7_Shield/PD_analyse', 'demo_xuanwuyanzheng_volumes_1.csv')
    df.to_csv(output_path, index=True)  # index=True 保留subject_id作为索引

    print(f"处理完成，共 {len(subjects)} 个被试，结果已保存到: {output_path}")
    print(f"检查缺失被试: {set(subjects) - set(df.index)}")

if __name__ == '__main__':
    main()
