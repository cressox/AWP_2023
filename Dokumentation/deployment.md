# Dokumentation - Fortschrittsbericht und Wissenschaftliche Erkenntnisse

### Untersuchung der Verwendung von Flask und React Native für die Bereitstellung von Anwendungen

In der anfänglichen Phase des Projektes wurde die Kombination von Flask und React Native für die Implementierung einer Hybrid-App in Betracht gezogen. Flask diente hierbei als Backend-Server, während React Native das Frontend-Framework bildete. Zur Unterstützung der Kommunikation zwischen der App und dem Backend war der Einsatz einer REST-API geplant. 

Flask wurde aufgrund seiner einfachen und flexiblen Implementierung einer REST-API ausgewählt, die eine effektive Interaktion mit der App ermöglicht. React Native hingegen wurde aufgrund seiner breiten Unterstützung für mobile Plattformen und seiner umfangreichen Entwicklergemeinschaft als Frontend-Framework gewählt.

Im Verlauf des Projekts änderten sich jedoch die Anforderungen, sodass der Einsatz eines Machine-Learning-Modells unwahrscheinlich wurde. Dies hatte auch Auswirkungen auf die Anforderungen an die Bereitstellung. Die initiale Kombination von Flask und React Native erfüllt diese neuen Anforderungen nicht mehr adäquat.

Folglich wurden alternative Technologien und Ansätze in Betracht gezogen, die besser zu den aktuellen Anforderungen passen. Im Folgenden werden der aktuelle Fortschrittsbericht und die wissenschaftlichen Erkenntnisse dargestellt, um eine fundierte Entscheidungsfindung zu ermöglichen.


## Fortschrittsbericht

Verschiedene Möglichkeiten und Optionen wurden im Rahmen der Untersuchungen betrachtet. Der folgende Abschnitt bietet einen Überblick über den aktuellen Fortschrittsbericht:

1. **Verwendung von React als Framework**: Die Nutzung von React als Framework für die Entwicklung der Anwendung wurde in Erwägung gezogen. In diesem Zusammenhang wurde die WebcamCapture-Komponente innerhalb von React erstellt, um Zugriff auf die Kamera des Geräts zu erhalten und Videoaufnahmen zu machen. React unterstützt effektiv die Erstellung interaktiver Benutzeroberflächen und ermöglicht eine strukturierte Komponenten-Codebildung.

2. **Erstellung einer Flask-API**: Ursprünglich wurde eine Flask-API vorgesehen, um die WebcamCapture-Komponente zu unterstützen und die Datenverarbeitung serverseitig durchzuführen. Allerdings wurde aufgrund von Geschwindigkeitsproblemen von dieser Lösung abgesehen.

3. **Kivy als Framework**: Das Kivy-Framework wurde ebenfalls zur Entwicklung der Anwendung in Betracht gezogen. Kivy, eine Open-Source Python-Bibliothek für die Entwicklung von Multitouch-Anwendungen, bietet Unterstützung für sowohl Android als auch iOS. Die Wahl von Kivy könnte den Entwicklungsprozess vereinfachen und die Notwendigkeit, eine separate API zur Kommunikation zwischen Frontend und Backend zu schreiben, eliminieren. Darüber hinaus könnte die Offline-Fähigkeit von Kivy in vorgeschlagenen Szenarien, in denen die Anwendung ohne kontinuierliche Internetverbindung funktionieren sollte, von besonderem Nutzen sein.

4. **Vergleich: Kivy vs. React Native mit Flask REST API**
Ein detaillierter Vergleich ist [hier](./kivy_vs_react.md) zu finden.

## Wissenschaftliche Erkenntnisse

Die Untersuchung lieferte mehrere wichtige Erkenntnisse:

1. **Leistungsoptimierung**: Der Verzicht auf eine API und die Ausführung des Codes clientseitig mit Pyscript könnte Potenzial zur Verbesserung der Leistung und Geschwindigkeit der Anwendung bieten. Eine direkte Datenverarbeitung auf dem Gerät ermöglicht eine schnellere Ausführung und reduziert Lat enzzeiten.

2. **Flexibilität**: Die Verwendung von React als Framework bietet Flexibilität bei der Entwicklung interaktiver Benutzeroberflächen und ermöglicht eine strukturierte Komponenten-Codebildung. Dies verbessert die Skalierbarkeit und Wartbarkeit der Anwendung.

3. **Harmonie von Pyscript und React**: Die Kompatibilität und Harmonie von Pyscript und React ist eine wichtige Fragestellung, die noch geklärt werden muss. Es bleibt ungewiss, wie effektiv Pyscript und React zusammenarbeiten und ob bei der Integration möglicherweise Herausforderungen auftreten könnten.

Auf der Grundlage dieser Erkenntnisse steht nun die Entscheidungsfindung, ob Pyscript eingesetzt werden kann und wie es optimal in eine App (egal ob webbasiert oder appbasiert) integriert werden kann, im Vordergrund. Die Untersuchungen und Tests werden fortgesetzt, um die Machbarkeit dieser Kombination zu ermitteln und eine fundierte Entscheidung zu treffen.
