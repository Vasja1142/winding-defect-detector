import os
import shutil
import math
import subprocess

def create_chunks(input_dir, output_base_dir, chunk_size=5000):
    print(f"--- Создание чанков для CVAT ---")
    print(f"Исходная директория: {input_dir}")
    print(f"Размер чанка: {chunk_size} изображений")

    # Убедимся, что obj.names существует
    obj_names_path = os.path.join(input_dir, "obj.names")
    if not os.path.exists(obj_names_path):
        print(f"Ошибка: Файл obj.names не найден в {input_dir}. Пожалуйста, убедитесь, что он там есть.")
        return

    # Получаем список всех .jpg файлов
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')])
    if not image_files:
        print(f"Ошибка: В директории {input_dir} не найдено изображений.")
        return

    print(f"Найдено {len(image_files)} изображений.")

    num_chunks = math.ceil(len(image_files) / chunk_size)
    print(f"Будет создано {num_chunks} чанков.")

    # Создаем базовую директорию для чанков
    os.makedirs(output_base_dir, exist_ok=True)

    for i in range(num_chunks):
        chunk_name = f"chunk_{i+1:02d}"
        chunk_dir = os.path.join(output_base_dir, chunk_name)
        os.makedirs(chunk_dir, exist_ok=True)

        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(image_files))
        current_chunk_files = image_files[start_index:end_index]

        print(f"\nОбработка чанка {i+1}/{num_chunks}: {chunk_name} ({len(current_chunk_files)} изображений)")

        # Копируем obj.names в директорию чанка
        shutil.copy(obj_names_path, os.path.join(chunk_dir, "obj.names"))

        for img_filename in current_chunk_files:
            base_name = os.path.splitext(img_filename)[0]
            label_filename = f"{base_name}.txt"

            src_img_path = os.path.join(input_dir, img_filename)
            src_label_path = os.path.join(input_dir, label_filename)

            dest_img_path = os.path.join(chunk_dir, img_filename)
            dest_label_path = os.path.join(chunk_dir, label_filename)

            shutil.copy(src_img_path, dest_img_path)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dest_label_path)
            else:
                # Если метки нет, создаем пустой файл (для негативных примеров)
                open(dest_label_path, 'a').close()

        # Запускаем конвертацию YOLO в CVAT XML для текущего чанка
        xml_output_file = os.path.join(chunk_dir, f"{chunk_name}.xml")
        command = [
            "python3", "/app/src/utils/yolo_to_cvat_xml.py",
            "--input_dir", chunk_dir,
            "--output_file", xml_output_file
        ]
        print(f"  -> Конвертация в CVAT XML для {chunk_name}...")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"  -> XML для {chunk_name} успешно создан: {xml_output_file}")
        except subprocess.CalledProcessError as e:
            print(f"  [ОШИБКА] Не удалось создать XML для {chunk_name}: {e.stderr}")
            print(f"  [ОШИБКА] Команда: {' '.join(command)}")
            print(f"  [ОШИБКА] Stdout: {e.stdout}")
            print(f"  [ОШИБКА] Stderr: {e.stderr}")

    print("\n--- Создание всех чанков и XML файлов завершено! ---")
    print(f"Чанки и XML файлы находятся в: {output_base_dir}")

if __name__ == '__main__':
    input_directory = "/app/data/06_prelabeled"
    output_base_directory = "/app/data/06_prelabeled/chunks"
    create_chunks(input_directory, output_base_directory)
