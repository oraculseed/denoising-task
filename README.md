# denoising-task
 Denoising task

Пример запуска
python denoising_task.py train --dataset dataset --k_train 10 --batch_size 512 --epochs 100 --augmentation 1


train - тренировка основной модели (удаление шума)
--dataset - путь к датасету
--k_train - кол-во обрабатываемых файлов в кадой директории диктора из датасета
--batch_size - размер батча
--epochs - кол-во эпох
--augmentation - 1/0 легкая аугментация (инверсия) 
 

train_classification - тренировка модели классификации clean/noisy
--dataset - путь к датасету
--k_train - кол-во обрабатываемых файлов в кадой директории диктора из датасета
--batch_size - размер батча
--epochs - кол-во эпох
 

predict - классификация и удаление шумов в файлах, создание отчета
--input - пусть к папке где лежат файлы и папки
--output - путь к папке куда будет сохранятся отчет и файлы

