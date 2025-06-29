# src/utils/remove_duplicate_boxes.py
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm

def find_and_remove_clones(input_xml_path, output_xml_path):
    """
    Читает XML-файл аннотаций CVAT, находит и удаляет дублирующиеся рамки.
    Дубликатом считается рамка с идентичными координатами и меткой.
    """
    print(f"--- Запуск операции 'Охота на Клонов' ---")
    print(f"Читаем исходный файл: {input_xml_path}")
    
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    
    # --- Шаг 1: Считаем, сколько раз встречается каждая уникальная рамка ---
    box_counts = {}
    total_boxes_before = 0
    
    for image in root.findall('image'):
        for box in image.findall('box'):
            total_boxes_before += 1
            # Создаем уникальный "отпечаток" рамки
            box_signature = (
                box.get('label'),
                box.get('xtl'),
                box.get('ytl'),
                box.get('xbr'),
                box.get('ybr')
            )
            box_counts[box_signature] = box_counts.get(box_signature, 0) + 1
            
    # --- Шаг 2: Находим "отпечаток" самой часто встречающейся рамки ---
    if not box_counts:
        print("В файле нет рамок. Нечего делать.")
        return

    most_common_box_signature = max(box_counts, key=box_counts.get)
    clone_count = box_counts[most_common_box_signature]
    
    print(f"\nОбнаружен 'супер-клон' (самая частая рамка):")
    print(f"  - Метка: {most_common_box_signature[0]}")
    print(f"  - Координаты: {most_common_box_signature[1:]}")
    print(f"  - Встречается раз: {clone_count}")
    
    if clone_count < 10: # Порог, чтобы случайно не удалить что-то нужное
        print("\nПредупреждение: Не найдено явных массовых дубликатов. Операция отменена.")
        return
        
    # --- Шаг 3: Создаем новый XML, удаляя все экземпляры "супер-клона" ---
    boxes_removed = 0
    for image in root.findall('image'):
        boxes_to_remove = []
        for box in image.findall('box'):
            current_signature = (
                box.get('label'),
                box.get('xtl'),
                box.get('ytl'),
                box.get('xbr'),
                box.get('ybr')
            )
            if current_signature == most_common_box_signature:
                boxes_to_remove.append(box)
                boxes_removed += 1
        
        for box in boxes_to_remove:
            image.remove(box)

    # --- Шаг 4: Сохраняем очищенный XML ---
    tree.write(output_xml_path)
    
    print(f"\n--- Операция Завершена! ---")
    print(f"Всего рамок было: {total_boxes_before}")
    print(f"Удалено 'клонов': {boxes_removed}")
    print(f"Осталось рамок: {total_boxes_before - boxes_removed}")
    print(f"Очищенный файл сохранен как: {output_xml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Удаляет массово дублирующиеся рамки из CVAT XML.")
    parser.add_argument('--input_xml', required=True, help='Путь к исходному XML файлу с ошибкой.')
    parser.add_argument('--output_xml', required=True, help='Путь для сохранения очищенного XML файла.')
    args = parser.parse_args()
    find_and_remove_clones(args.input_xml, args.output_xml)