# src/yolo_to_cvat_xml.py
import os
import sys
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm

def yolo_to_cvat_xml(input_dir, output_file):
    """
    Конвертирует датасет в формате YOLO в один XML-файл формата CVAT 1.1.
    """
    print("--- Запуск Конвертера из YOLO в CVAT XML ---")
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"[ОШИБКА] Директория не найдена: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Читаем имена классов
    names_file = input_path / "obj.names"
    if not names_file.exists():
        print(f"[ОШИБКА] Файл с именами классов 'obj.names' не найден!", file=sys.stderr)
        sys.exit(1)
    
    idx_to_class = {i: name.strip() for i, name in enumerate(names_file.read_text().splitlines())}
    print(f"1. Найдены классы: {list(idx_to_class.values())}")

    # 2. Собираем XML структуру
    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<annotations>',
        '  <version>1.1</version>',
        '  <meta>',
        '    <task>',
        '      <labels>'
    ]
    for idx, name in idx_to_class.items():
        xml_lines.append(f'        <label><name>{name}</name><attributes></attributes></label>')
    xml_lines.extend(['      </labels>', '    </task>', '  </meta>'])
    
    image_files = sorted([f for f in input_path.glob('*.jpg')])
    print(f"2. Найдено {len(image_files)} изображений для обработки.")
    
    image_id = 0
    for image_path in tqdm(image_files, desc="Конвертация"):
        # Получаем размеры изображения
        try:
            img = cv2.imread(str(image_path))
            height, width, _ = img.shape
        except Exception as e:
            print(f"\n[ПРЕДУПРЕЖДЕНИЕ] Не удалось прочитать изображение {image_path.name}: {e}")
            continue

        # Формируем XML блок для изображения
        xml_lines.append(f'  <image id="{image_id}" name="{image_path.name}" width="{width}" height="{height}">')

        # Ищем соответствующий .txt файл
        label_path = image_path.with_suffix('.txt')
        if label_path.exists():
            annotations = label_path.read_text().splitlines()
            for ann in annotations:
                parts = ann.split()
                if len(parts) != 5: continue
                
                class_idx, x_c, y_c, w, h = map(float, parts)
                
                # Конвертируем YOLO в абсолютные координаты (xtl, ytl, xbr, ybr)
                box_w = w * width
                box_h = h * height
                xtl = (x_c * width) - (box_w / 2)
                ytl = (y_c * height) - (box_h / 2)
                xbr = xtl + box_w
                ybr = ytl + box_h
                
                label_name = idx_to_class.get(int(class_idx), "unknown")
                
                xml_lines.append(f'    <box label="{label_name}" occluded="0" source="manual" xtl="{xtl:.2f}" ytl="{ytl:.2f}" xbr="{xbr:.2f}" ybr="{ybr:.2f}" z_order="0"></box>')
        
        xml_lines.append('  </image>')
        image_id += 1
        
    xml_lines.append('</annotations>')
    
    # 3. Сохраняем результат
    output_path = Path(output_file)
    output_path.write_text("\n".join(xml_lines))
    print(f"\n3. УСПЕХ! Файл аннотаций сохранен в: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Конвертер из YOLO в CVAT XML 1.1.")
    parser.add_argument('--input_dir', required=True, type=str, help='Папка с датасетом YOLO (jpg, txt, obj.names).')
    parser.add_argument('--output_file', required=True, type=str, help='Путь для сохранения итогового XML файла.')
    args = parser.parse_args()
    yolo_to_cvat_xml(args.input_dir, args.output_file)