import sqlite3
import pandas as pd
import sys
import glob
import os

def get_table_columns(cursor, table_name):
    """获取指定表的列名列表"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        return [info[1] for info in cursor.fetchall()]
    except Exception:
        return []

def parse_nsys_sqlite(result_dir):
    # 1. 寻找 sqlite 文件
    search_pattern = os.path.join(result_dir, "*.sqlite")
    files = glob.glob(search_pattern)
    if not files:
        print(f"[Error] No .sqlite file found in {result_dir}")
        return

    db_file = files[0]
    print(f"Reading database: {db_file}")

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # 2. 检查存在的表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        # === 动态构建查询逻辑 ===
        data_table = "GPU_METRICS"
        info_table = "TARGET_INFO_GPU_METRICS"
        
        if data_table not in tables:
            print(f"[Error] Critical table {data_table} not found!")
            return

        if info_table not in tables:
            print(f"[Error] Critical table {info_table} not found!")
            return

        # 3. 智能分析列名
        data_cols = get_table_columns(cursor, data_table)
        info_cols = get_table_columns(cursor, info_table)

        print(f"Debug: {data_table} columns: {data_cols}")
        print(f"Debug: {info_table} columns: {info_cols}")

        # 寻找连接键 (Join Key)
        # 通常是 metricId, typeId, 或 id
        join_key_data = next((col for col in ['metricId', 'typeId', 'id'] if col in data_cols), None)
        join_key_info = next((col for col in ['metricId', 'typeId', 'id'] if col in info_cols), None)

        # 寻找名称列 (Name Column)
        name_col_info = next((col for col in ['name', 'metricName', 'label', 'description'] if col in info_cols), None)

        if not join_key_data or not join_key_info or not name_col_info:
            print("[Error] Could not identify Join Keys or Name columns automatically.")
            return

        print(f"Constructing Query: JOIN ON {data_table}.{join_key_data} = {info_table}.{join_key_info}")

        # 4. 构建并执行 SQL
        query = f"""
        SELECT 
            gm.timestamp AS timestamp_ns,
            ti.{name_col_info} AS metric_name,
            gm.value
        FROM {data_table} gm
        JOIN {info_table} ti ON gm.{join_key_data} = ti.{join_key_info}
        ORDER BY gm.timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print("[Warning] Query executed but returned no data.")
            return

        # 5. 数据处理与导出
        start_ns = df['timestamp_ns'].min()
        df['Time_s'] = (df['timestamp_ns'] - start_ns) / 1e9
        
        # 透视表: 把 [Time, Metric, Value] 变成 [Time, Power, SM_Util...]
        pivot_df = df.pivot_table(index='Time_s', columns='metric_name', values='value')
        
        output_csv = os.path.join(result_dir, "final_gpu_metrics.csv")
        pivot_df.to_csv(output_csv)
        print(f"\n[Success] Metrics extracted to: {output_csv}")
        print("Columns extracted:", pivot_df.columns.tolist())
        print("You can now plot this CSV to see the parallel power sequence!")

    except Exception as e:
        print(f"[Error] Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python sqlite_csv_v3.py <results_dir>")
    # else:
    parse_nsys_sqlite(sys.argv[1])