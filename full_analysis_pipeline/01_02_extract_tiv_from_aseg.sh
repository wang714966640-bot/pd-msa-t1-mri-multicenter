#!/bin/bash

# 定义输入文件和数据路径
SUBJECTS_FILE="/mnt/data/download_yanzheng_clinical2/sublist.txt"
BASE_DIR="/mnt/data/download_yanzheng_clinical2"
OUTPUT_FILE="eTIV_summary_yanzheng_clinical2.csv"

# 检查输入文件是否存在
if [ ! -f "$SUBJECTS_FILE" ]; then
    echo "错误: 文件 $SUBJECTS_FILE 不存在！"
    exit 1
fi

# 创建CSV表头
echo "SubjectID,eTIV(mm3)" > "$OUTPUT_FILE"

# 逐行读取被试ID并提取eTIV
while IFS= read -r subject_id; do
    subject_dir="$BASE_DIR/$subject_id"
    if [ ! -d "$subject_dir" ]; then
        echo "警告: 被试目录 $subject_dir 不存在，跳过..."
        continue
    fi

    aseg_file="$subject_dir/stats/aseg.stats"
    if [ ! -f "$aseg_file" ]; then
        echo "警告: $aseg_file 不存在，跳过..."
        continue
    fi

    # 修正提取逻辑：使用更精确的匹配方式
    eTIV=$(grep "Measure EstimatedTotalIntraCranialVol" "$aseg_file" | awk -F', ' '{print $4}' | awk '{print $1}')
    
    if [ -n "$eTIV" ]; then
        echo "$subject_id,$eTIV" >> "$OUTPUT_FILE"
        echo "已提取: $subject_id -> eTIV = $eTIV mm³"
    else
        echo "警告: 无法从 $aseg_file 提取eTIV，跳过..."
    fi
done < "$SUBJECTS_FILE"

echo "处理完成！结果已保存到 $OUTPUT_FILE"
