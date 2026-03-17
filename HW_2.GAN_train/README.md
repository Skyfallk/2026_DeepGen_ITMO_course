# ДЗ 1. Имплементация GAN

1. Скачать датасет [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (можно через [pytorch](https://pytorch.org/vision/stable/generated/torchvision.datasets.CelebA.html))
2. Имплементировать CSPup блок (5 баллов)
![CSPup блок](imgs/csp_up.jpg)
3. Имплементировать генератор GAN по заданной архитектурной схеме (10 баллов)
![Архитектура генератора](imgs/generator.jpg)
4. Обучить имплементированный GAN (5 баллов)
5. Добиться сходимости (регуляризации, изменение архитектуры, фишки с train loop) (10 баллов)

## Кириллов, доп. задание:

Реализация mapper.

1. Создать класс PixelNorm для нормализации вектора z
- наследуется от torch.nn.Module
- при инициализаци фиксирует eps=1e-8 
- нормализует сигнал ветокра x / sqrt(mean(x^2) + eps)
2.  Создать класс MappingNet
- наследуется от torch.nn.Module
- два обязательных метода: init и forward
- входные параметры: 
z_dim — размер входного шума
w_dim — размер выходного вектора
num_layers — глубина сети
3. Собрать MLP внутри класса
- с помощью torch.nn.Sequential собрать последовательность слоев (num_layers штук)
- каждый слой это одинаковый Linear + LeakyRelU(0.2)
- конечный выходной слой должен иметь размерность w_dim (обычно z_dim=w_dim)
4. Реализовать forward
z -> PixelNorm -> MLP -> w
5. Добавить mapper в train loop
- маппер между z и G
- добавить mapper в оптимизатор
6. Сравнить результаты с экспериментами без маппера
