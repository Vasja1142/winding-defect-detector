# src/utils/smart_merge.py
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def smart_merge_classes(input_dir, output_dir, keep_label_name='row_gap'):
    """
    Объединяет все классы в один ('defect'), кроме одного указанного.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Считываем оригинальную карту классов ---
    names_file = input_path / "obj.names"
    if not names_file.exists():
        print(f"Ошибка: не найден файл {names_file}")
        return
        
    original_names = {name.strip(): i for i, name in enumerate(names_file.read_text().splitlines())}
    if keep_label_name not in original_names:
        print(f"Ошибка: класс '{keep_label_name}' не найден в исходных данных.")
        return
        
    keep_class_id_original = original_names[keep_label_name]
    
    # --- Создаем новую карту классов ---
    # 0: keep_label_name
    # 1: 'defect'
    new_names = {0: keep_label_name, 1: 'defect'}
    print(f"Новая карта классов: {new_names}")
    
    # --- Обрабатываем файлы ---
    all_files = list(input_path.glob('*.jpg'))
    for image_file in tqdm(all_files, desc="Умное слияние"):
        # Копируем изображение
        os.system(f'cp "{image_file}" "{output_path / image_file.name}"')
        
        # Обрабатываем файл разметки
        label_file = image_file.with_suffix('.txt')
        if not label_file.exists(): continue
            
        new_lines = []
        lines = label_file.read_text().splitlines()
        for line in lines:
            parts = line.split()
            if not parts: continue
            
            original_id = int(parts[0])
            
            if original_id == keep_class_id_original:
                # Это наш особый класс. Присваиваем ему новый ID = 0
                parts[0] = '0'
            else:
                # Это все остальные классы. Присваиваем им ID = 1
                parts[0] = '1'
            
            new_lines.append(" ".join(parts))
            
        (output_path / label_file.name).write_text("\n".join(new_lines))

    # --- Сохраняем новый obj.names ---
    (output_path / "obj.names").write_text(f"{keep_label_name}\ndefect\n")

    print("\n--- Умное слияние завершено! ---")
    print(f"Новый двухклассовый датасет сохранен в: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    smart_merge_classes(args.input_dir, args.output_dir)