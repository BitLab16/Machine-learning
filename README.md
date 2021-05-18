# Machine-learning

[![Python CI](https://github.com/BitLab16/Machine-learning/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/BitLab16/Machine-learning/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=BitLab16_Machine-learning&metric=alert_status)](https://sonarcloud.io/dashboard?id=BitLab16_Machine-learning)

Per l'esecuzione dell'applicativo:
1) Installare Python (versione <=3.8)
2) Installare Pip:
    a) py -m pip install --upgrade pip               (Windows)
    b) python3 -m pip install --user --upgrade pip   (Linux/Mac OS)
3) Creare una cartella in cui si vuole creare il virtual environement
4) Posizionarsi all'interno della cartella ed eseguire: 
    a) py -m venv env      (Windows)  
    b) python3 -m venv env (Linux/Mac OS)
5) Posizionarsi all'interno di nome_cartella ed eseguire: 
    a) env\Scripts\activate     (Windows)
    b) source env/bin/activate  (Linux/Mac OS)
6) Successivamente, eseguire: pip install -r requirements.txt
7) Eseguire: flask run
