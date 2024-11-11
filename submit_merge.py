import pandas as pd

# 读取两个CSV文件
df_a = pd.read_csv('/home/ccip/data/wangyuanxiang/xyy/reproduce/MEIJU2025-baseline-master/submission-whisper-0.5882.csv')
df_b = pd.read_csv('/home/ccip/data/wangyuanxiang/xyy/reproduce/MEIJU2025-baseline-master/submission-intent0.6048.csv')

# 合并两个DataFrame，按filename列
merged_df = pd.merge(df_a[['filename', 'emo_pred']], df_b[['filename', 'int_pred']], on='filename', how='inner')

# 保存合并后的结果到一个新的CSV文件
merged_df.to_csv('merged_output_val_emo0.583_int0.6048.csv', index=False)

# 打印合并后的数据
print(merged_df)
