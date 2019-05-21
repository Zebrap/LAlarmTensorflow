# LAlarmTensorflow

Sieć neuronowa tworząca model tensorflow.
Wymagania Pythona 3.5.4 w wersji 64-bitowej dla Windows ze strony https://www.python.org/downloads/ a następnie wykonać polecenia:
pip install tensorflow
pip install keras

Trenowanie sieci: python index.py
Testowanie sieci: python classify.py
alarm2.csv - przykładowe dane do uczenia sieci struktura pliku: czas, wibracje, głośność, dźwięk, ocena (przeskalowana do zakresu 0 - 1)
