# Dokumentation - Arbeitsfortschritt und Erkenntnisse - Meilenstein 1

## zu der Entwicklung eines Algorithmus für die Detektion der Müdigkeit

Der erste Teil der Entwicklung bestand aus der Recherche zu möglichen Vorgehensweisen und der Bewertung dieser. Dabei war das Ziel folgende Fragen zu klären:

1. Wie detektiere ich ein Gesicht und das Auge?
2. Wie erkenne ich Landmarks in einem Gesicht?
3. Wie lässt sich Blinzeln detektieren?
4. Welche Features zur Detektion von Müdigkeit gibt es?
5. Wie lässt sich anhand der Features die Müdigkeit detektieren?

## Arbeitsfortschritt

Für die Detektion von Müdigkeit im Straßenverkehr gibt es verschiedene Möglichkeiten. Diese lassen sich in drei Ansätze unterteilen (Ramzan et al. [Ramzan2019])

1. Fahrweise: Aufgrund der Analyse der Fahrweise lässt sich Unaufmerksamkeit durch Müdigkeit ermitteln. Ein Beispiel wäre das Halten (Wach) oder Verlassen (Müde) der Fahrspur.
2. Physiologisch: Über Physiologische Daten, wie der Herzfrequenz oder der Atemfrequenz, wird die Müdigkeit detektiert. Eine niedrige Herzfrequenz weist beispielsweise auf einen entspannten, müden Zustand hin
3. Verhaltensweise: Das Verhalten eines Fahrers sagt viel über den Müdigkeitszustand aus. So sind Anzeichen für Müdigkeit Gähnen oder ein verändertes Blinzelverhalten.

Bei unserer Arbeit wird das Verhalten des Fahrers analysiert, wobei der Fokus auf dem Auge liegt. Zunächst ist dabei die Detektion des Auges und das Erkennen von Landmarks ein wichtiger Schritt, um darauf folgend Features zu extrahieren. Mit den Features lässt sich der Müdigkeitsstatus eines Fahrers ermitteln.

Dabei hängt die Gesichts- und Landmarkdetektion eng zusammen. Denn die Wahl des Algorithmus für die Gesichtsdetektion ist abhängig von dem Hintergrund der Detektion. Einen guten Überblick bietet Kumar et al. [Kum2018]. In unserem Fall möchten wir Landmarks in einem Gesicht finden, wofür sich der Viola-Jones Algorithmus eignet. Für die Detektion von Gesichern zur Müdigkeitsdetektion wurde dieser Algorithmus vielfach verwendet (Ghoddoosian et al. [Gho2019], Arunasalam et al [Aru2020]).

Eine gute Implementierung des Viola-Jones Algorithmus bietet der Dlib Face Detector. Dabei wurde der Viola-Jones Algorithmus angepasst und erweitert, um eine Realzeitanwendung zu relaisieren. Genauere Infos zu dem Algorithmus finden sich hier: <http://dlib.net/face_landmark_detection.py.html>
Eine andere Mögichkeit wäre der MediaPipe Face Detector, welcher jedoch schwieriger zu implementieren ist. Ein Vorteil des Dlib Face Detectors ist, dass das Modell vortrainiert wurde und offline anwendbar ist. Das Modell gibt 68 Landmarks im Gesicht, wovon es 6 Landmarks pro Auge gibt. Diese reichen aus, um Features zu extrahieren, wie auch von Ghoddoosian et al. [Gho2019].

Ein grundlegendes  Feature, auf welche weitere Features aufbauen ist der EAR (Eye Aspect Ratio) Wert. Der Wert berechnet das Verhältnis zwischen den Augenliedern, um die Augenöffnung zu bestimmen. Dies ist ein wichtiger Indikator für den Müdugkeitsstatus. Darauffolgend oder auch unabhängig vom EAR-Wert lassen sich viele weitere Features berechnen. Eine Liste an möglichen Features bietet
Ebrahim et al. [Ebr2016]. Auch entscheidend ist die Frequenz des Blinzelns (PERCLOS-Wert) Bei Müdigkeit verändert sich das Blinzelverhalten. Bei Dreißig et al. [Dre2020] spielt dieser Wert eine entscheidende Rolle bei der Müdigkeitsdetektion.

Der nächste Schritt ist die Einteilung in Wach/Müde auf Grundlage der ausgewählten Features. Hier gibt es viele verschiedene Möglichkeiten, wie [Ramzan2019] vorstellt. Es wird entweder ein Hidden Markov Model (von Ghoddoosian et al. [Gho2019] verwendet), ein Convolutional Neural Network oder eine Support Vextor Machine verwendet. Jedes Model hat Vor- und Nachteile, wobei das Hidden Markov Model vermutlich die beste Wahl für eine schnelle Anwendung ist und die wenigsten Nachteile bietet. Dies muss aber noch weiter analysiert werden.

Vor der Klassifikation mittels der Features müssen diese vorher Vorverarbeitet werden, um individuelle Unterschiede herauszufiltern. Dies erfolgt durch eine Normalisierung der Features, wie in Ghoddoosian et al. [Gho2019] beschrieben. Zudem ist eine Kalibrierung bei Aufnahme einer neuen Person wichtig, um Individualitäten zu berücksichtigen. Auch hier muss das weitere Vorgehen noch weiter bestimmt werden.

## Erkentnisse

Wie sich aus der Rechnerche ergeben hat sind einige Mehtoden zur Gesichsdetektion in der Praxis für Realzeitanwendungen nicht geeignet, da diese zu lange Berechnungszeit brauchen. Auch gibt es nicht viele Gesichtserkennungsmethoden, welche für die eigene Implementierung genutzt werden können. Viele Gesichtserkennungsmethoden müssen auch bezahlt werden. Diese sind meistens noch genauer und bieten mehr Landmarks, was für unsere ANwendung jedoch gar nicht nötig ist, denn 6 Landmarks pro Auge reichen vollkommen aus.

Es gibt viele Kleinigkeiten zu beachten. Insbesondere die Vorverarbeitung der Feaures war vorher kein Punkt, welchen wir betrachtet haben, dieser erscheint jedoch wie im Arbeitsfortschritt beschrieben, unerlässlich.

## Folgende Arbeitsschritte

Auf Basis des bisherigen Arbeitsfortschritts sind nun folgende Punkte zu bearbeiten:

1. Exktraktion der entscheidenden Features aus der großen Auswahl an möglichen Features
2. Bestimmung des Vorgehens bei der Vorverarbeitung
3. Auswahl des besten Klassifikators nach den Aspekten Implementierbarkeit und Performance
4. Auswahl geeigneter Datensätze zum Trainieren des Klassifikators
5. Implementierung des Dlib Face Detectors zur Detektion von Landmarks im Gesicht

## Literatur

[Aru2020] Arunasalam, M.; Yaakob, N.; Amir, A.; Elshaikh, M.; Azahar, N. F. (2020): Real-Time Drowsiness Detection System for Driver Monitoring. In: IOP Conf. Ser.: Mater. Sci. Eng. 767 (1), S. 12066. DOI: 10.1088/1757-899X/767/1/012066

[Gho2019] Ghoddoosian, Reza; Galib, Marnim; Athitsos, Vassilis (2019): A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection. Online verfügbar unter <http://arxiv.org/pdf/1904.07312v1>

[Ramzan2019] Ramzan, Muhammad; Khan, Hikmat Ullah; Awan, Shahid Mahmood; Ismail, Amina; Ilyas, Mahwish; Mahmood, Ahsan (2019): A Survey on State-of-the-Art Drowsiness Detection Techniques. In: IEEE Access 7, S. 61904–61919. DOI: 10.1109/ACCESS.2019.2914373

[Kum2018] Kumar, Ashu; Kaur, Amandeep; Kumar, Munish (2019): Face detection techniqeues: a review. In: Artif Intell Rev 52 (2), S. 927–948. DOI: 10.1007/s10462-018-9650-2

[Ebr2016] Parisa Ebrahim (2016): Driver drowsiness monitoring using eye movement features derived from electrooculography. Doktorarbeit.

[Dre2020] Mariella Dreißig, Mohamed Hedi Baccour, Tim Schäck, Enkelejda Kasneci (2020): Driver Drowsiness Classification Based on Eye Blink and Head Movement Features Using the k-NN Algorithm. 

