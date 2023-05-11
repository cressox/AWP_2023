# Dokumentation - Arbeitsfortschritt und Erkenntnisse

### Verwendung von Flask und React Native für das Deployment

Eine der ursprünglichen Ideen für das Deployment war die Verwendung von Flask als Backend-Server und React Native als Frontend-Framework für die Entwicklung einer Hybrid-App. Hierbei sollte eine REST-API verwendet werden, um die Kommunikation zwischen der App und dem Backend zu ermöglichen. 

Flask wurde als Backend-Server gewählt, da es eine einfache und flexible Möglichkeit bietet, eine REST-API zu implementieren und mit der App zu kommunizieren. React Native wurde als Frontend-Framework gewählt, da es eine breite Unterstützung für mobile Plattformen bietet und eine große Entwicklercommunity hat.

Allerdings haben sich die Anforderungen im Laufe der Entwicklung geändert und es ist sehr wahrscheinlich, dass wir kein Machine Learning-Modell mehr verwenden. Dadurch ergeben sich auch andere Anforderungen an das Deployment. Die ursprüngliche Idee mit Flask und React Native wird den neuen Anforderungen nicht mehr gerecht.

Ich werde daher alternative Ansätze und Technologien in Betracht ziehen, die besser zu den aktuellen Anforderungen passen. Im Folgenden werde ich den aktuellen Arbeitsfortschritt und die Erkenntnisse dokumentieren, um eine fundierte Entscheidung zu treffen.


## Arbeitsfortschritt

Im Laufe meiner Recherche zum Thema habe ich verschiedene Möglichkeiten und Optionen in Betracht gezogen. Hier ist ein Überblick über meinen bisherigen Arbeitsfortschritt:

1. **React als Framework**: Ich habe in Betracht gezogen, React als Framework zu nutzen, um meine Anwendung zu entwickeln. Dabei habe ich die WebcamCapture-Komponente innerhalb von React entwickelt, um auf die Kamera des Geräts zuzugreifen und Videoaufnahmen zu machen. React bietet eine gute Unterstützung für die Erstellung interaktiver Benutzeroberflächen und ermöglicht es mir, den Code in Komponenten zu strukturieren.

2. **Flask-API**: Ursprünglich hatte ich die Idee, eine Flask-API zu erstellen, um die WebcamCapture-Komponente zu unterstützen und die Datenverarbeitung auf einem Server durchzuführen. Allerdings habe ich mich aufgrund der Geschwindigkeitsprobleme gegen diese Lösung entschieden.

3. **Pyscript und React**: Anschließend habe ich mich mit Pyscript beschäftigt und versucht herauszufinden, ob und wie gut es mit React harmoniert. Dabei stehe ich jedoch vor der Herausforderung, zu entscheiden, ob Pyscript die richtige Wahl ist und wie ich es in eine App (egal ob webbasiert oder appbasiert) einbetten kann.

## Erkenntnisse

Im Verlauf meiner Recherche habe ich einige wichtige Erkenntnisse gewonnen:

1. **Leistungsoptimierung**: Der Verzicht auf eine API und die Ausführung des Codes clientseitig mit Pyscript bieten Potenzial zur Verbesserung der Leistung und Geschwindigkeit meiner Anwendung. Die direkte Datenverarbeitung auf dem Endgerät ermöglicht eine schnellere Ausführung und reduziert Latenzzeiten.

2. **Flexibilität**: Die Verwendung von React als Framework bietet mir die Flexibilität, interaktive Benutzeroberflächen zu entwickeln und den Code in Komponenten zu strukturieren. Dadurch wird die Skalierbarkeit und Wartbarkeit meiner Anwendung verbessert.

3. **Harmonie von Pyscript und React**: Die Kompatibilität und Harmonie von Pyscript und React ist eine wichtige Fragestellung, die noch geklärt werden muss. Es ist unklar, wie gut Pyscript und React zusammenarbeiten und ob möglicherweise Herausforderungen bei der Integration auftreten.

Basierend auf meinen Erkenntnissen stehe ich nun vor der Herausforderung, zu entscheiden, ob ich Pyscript nutzen kann und wie ich es am besten in eine App (egal ob webbasiert oder appbasiert) einbetten kann. Ich werde weiterhin meine Forschung und Tests durchführen, um die Machbarkeit dieser Kombination zu ermitteln und eine fundierte Entscheidung zu treffen.
