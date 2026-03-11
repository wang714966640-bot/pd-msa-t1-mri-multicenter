#!/bin/bash

# 简单有效的TIV提取脚本
# 基于实际文件夹结构分析

# 配置路径
BASE_DIR="/mnt/data/download_yanzheng_clinical2"
SUBJECT_LIST="${BASE_DIR}/sublist.txt"
OUTPUT_FILE="${BASE_DIR}/TIV_extracted.csv"

echo "开始提取TIV数据..."

# 创建输出文件并写入表头
echo "SubjectID,TIV_mm3" > "$OUTPUT_FILE"

# 统计变量
total_subjects=0
successful_extractions=0

# 逐行读取被试列表
while IFS= read -r subject_id || [[ -n "$subject_id" ]]; do
    # 去除空格
    subject_id=$(echo "$subject_id" | xargs)
    
    # 跳过空行
    if [ -z "$subject_id" ]; then
        continue
    fi
    
    total_subjects=$((total_subjects + 1))
    subject_dir="${BASE_DIR}/${subject_id}"
    
    echo "处理被试: $subject_id"
    
    # 检查被试目录是否存在
    if [ ! -d "$subject_dir" ]; then
        echo "  警告: 目录不存在 - $subject_dir"
        echo "$subject_id,DirectoryNotFound" >> "$OUTPUT_FILE"
        continue
    fi
    
    # 方法1: 尝试从synthseg.vol.csv提取 (推荐方法)
    synthseg_file="${subject_dir}/stats/synthseg.vol.csv"
    if [ -f "$synthseg_file" ]; then
        # 从CSV文件提取TIV值 (第2行第2列)
        tiv_value=$(awk -F',' 'NR==2 {print $2}' "$synthseg_file" 2>/dev/null)
        
        if [ -n "$tiv_value" ] && [ "$tiv_value" != "" ]; then
            echo "$subject_id,$tiv_value" >> "$OUTPUT_FILE"
            echo "  成功: TIV = $tiv_value mm³ (来自synthseg.vol.csv)"
            successful_extractions=$((successful_extractions + 1))
            continue
        fi
    fi
    
    # 方法2: 尝试从brainvol.stats提取 (备用方法)
    brainvol_file="${subject_dir}/stats/brainvol.stats"
    if [ -f "$brainvol_file" ]; then
        # 从stats文件提取Mask Volume (最接近TIV的指标)
        tiv_value=$(grep "Mask Volume" "$brainvol_file" | awk -F', ' '{print $4}' | awk '{print $1}' 2>/dev/null)
        
        if [ -n "$tiv_value" ] && [ "$tiv_value" != "" ]; then
            echo "$subject_id,$tiv_value" >> "$OUTPUT_FILE"
            echo "  成功: TIV = $tiv_value mm³ (来自brainvol.stats)"
            successful_extractions=$((successful_extractions + 1))
            continue
        fi
    fi
    
    # 如果两种方法都失败
    echo "  警告: 无法提取TIV数据"
    echo "$subject_id,ExtractionFailed" >> "$OUTPUT_FILE"

done < "$SUBJECT_LIST"

echo ""
echo "=== 提取完成 ==="
echo "总被试数: $total_subjects"
echo "成功提取: $successful_extractions"
echo "失败数量: $((total_subjects - successful_extractions))"
echo "结果文件: $OUTPUT_FILE"

# 显示前几行结果
echo ""
echo "=== 结果预览 ==="
head -10 "$OUTPUT_FILE"
