import os

# 出力するファイル名
OUTPUT_FILENAME = "project_contents.txt"

# 除外する拡張子（メディアファイル等）
IGNORE_EXTENSIONS = {
    # 画像
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico', '.svg', '.webp',
    # 音声
    '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma',
    # 動画
    '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v',
    # その他バイナリや不要ファイル
    '.exe', '.dll', '.so', '.o', '.pyc', '.zip', '.tar', '.gz', '.7z', '.pdf','.tres',
    '.ds_store','.uid','.import','.mtl','.obj','.tscn',
}
# 除外するディレクトリ名（探索もしない）
IGNORE_DIRS = {
    '.git', '.idea', '.vscode', '__pycache__', 'node_modules', 
    'venv', '.venv', 'build', 'dist', 'target', '.next', '.nuxt','shader','addons',
}

def main():
    # スクリプトを実行したディレクトリをプロジェクトのルートとする
    root_dir = os.getcwd()
    
    output_path = os.path.join(root_dir, OUTPUT_FILENAME)

    print(f"Project Root: {root_dir}")
    print("Processing...")

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for current_root, dirs, files in os.walk(root_dir):
                # 除外ディレクトリを探索リストから削除（in-place変更）
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

                for file in files:
                    # 出力ファイル自身は読み込まない
                    if file == OUTPUT_FILENAME:
                        continue
                    
                    # 自分自身（このスクリプト）も除外する場合は以下を有効化
                    # if file == os.path.basename(__file__): continue

                    file_path = os.path.join(current_root, file)
                    _, ext = os.path.splitext(file)
                    
                    # 1. 拡張子チェック
                    if ext.lower() in IGNORE_EXTENSIONS:
                        # print(f"[Skip Media] {file}") # ログが多すぎる場合はコメントアウト
                        continue

                    # 2. テキストファイルとしての読み込み試行
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                        
                        # 3. 相対パスの取得と整形
                        relative_path = os.path.relpath(file_path, root_dir)
                        # Windowsのバックスラッシュをスラッシュに統一（好みで変更可）
                        relative_path = relative_path.replace(os.sep, '/')
                        
                        # 書き込み
                        outfile.write(f"\n\n-------------------------------------\n\n--- START OF FILE [{relative_path}] ---\n\n\n")
                        outfile.write(content)
                        outfile.write("\n\n") # ファイル間の区切り
                        
                        print(f"[Added] {relative_path}")

                    except UnicodeDecodeError:
                        # UTF-8で読めないファイル（バイナリ等）はスキップ
                        print(f"[Skip Binary] {file}")
                    except Exception as e:
                        print(f"[Error] {file}: {e}")

        print(f"\n完了しました。内容は '{OUTPUT_FILENAME}' に保存されました。")

    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()