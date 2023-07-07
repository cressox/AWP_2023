# Vergleichsstudie: Kivy gegenüber React Native mit Flask REST API

Im Kontext der bisherigen Forschungsarbeiten und der vorgegebenen Entwicklungsumgebung werden im Folgenden Überlegungen zur Gegenüberstellung zwischen dem Einsatz von Kivy und der Kombination aus React Native mit einer Flask REST API für die Entwicklung einer Anwendung zur Erkennung von Müdigkeit präsentiert.

## Kivy

Kivy stellt ein Open-Source-Python-Framework für die Erstellung von Multitouch-Applikationen dar. Es ermöglicht plattformübergreifende Entwicklungen (Linux/OS X/Windows/Android/iOS) und steht unter der MIT-Lizenz. Insbesondere für Anwendungen, die Multitouch, Gesten und andere moderne Touch-Features benötigen, ist Kivy gut geeignet.

**Positive Aspekte**:
- Kivy ist in Python verfasst und ermöglicht damit die einheitliche Verwendung einer einzigen Sprache für das gesamte Projekt.
- Mit Kivy lässt sich die Anwendung als eigenständige App auf Android und iOS bereitstellen. Dies hat zur Folge, dass der Benutzer keine zusätzliche Software installieren muss, um die App nutzen zu können.
- Da Kivy auf OpenGL ES 2 basiert, bietet es eine beträchtliche Leistungsfähigkeit für grafikintensive Applikationen.
- Kivy bietet die Option offline zu arbeiten, was für die Anwendung zur Erkennung von Müdigkeit besonders wertvoll ist.

**Negative Aspekte**:
- Im Vergleich zu anderen Frameworks ist Kivy nicht so weit verbreitet, was sich in weniger Ressourcen, weniger aktualisierter Dokumentation und einer kleineren Community, die bei Problemen unterstützt, niederschlägt.
- Trotz der Tatsache, dass Kivy einige native Look-and-Feel-Widgets bereitstellt, könnte es eine Herausforderung darstellen, das Aussehen und Verhalten der App so anzupassen, dass sie den Erwartungen der Nutzer auf unterschiedlichen Plattformen gerecht wird.
- Generell ist Python langsamer als native Sprachen wie Java für Android und Swift für iOS. Daher könnte die Leistung, insbesondere bei rechenintensiven Aufgaben, ein Problem darstellen.

## React Native mit Flask REST API

React Native ist ein Framework zur Erstellung von nativen Apps für Android und iOS in JavaScript. Es basiert auf React, der JavaScript-Bibliothek von Facebook für das Aufbau von Benutzeroberflächen, zielt aber im Gegensatz zu React auf mobile Plattformen ab.

**Positive Aspekte**:
- Mit React Native ist es möglich, in JavaScript zu programmieren, einer der bekanntesten und weit verbreitesten Programmiersprachen, und dabei gleichzeitig native Leistung und ein natives Look-and-Feel zu erreichen.
- Im Vergleich zu Kivy verfügt React Native über eine deutlich größere Community und mehr Ressourcen, was die Suche nach Hilfe und Lösungen für auftretende Probleme erleichtert.
- Die Kombination mit einer Flask REST API ermöglicht es, das Backend in Python und das Frontend mit React Native zu gestalten. Dies könnte mehr Flexibilität hinsichtlich der Strukturierung des Codes bieten und die Möglichkeit schaffen, bewährte Praktiken und Muster aus beiden Welten zu nutzen.

**Negative Aspekte**:
- Bei der Verwendung einer REST API besteht die Notwendigkeit einer ständigen Internetverbindung, um mit dem Backend interagieren zu können. Dies könnte für die Anwendung zur Erkennung von Müdigkeit ein Problem darstellen, da sie auch offline funktionieren muss.
- Es könnte zu Schwierigkeiten kommen, die rechenintensiven Teile der Anwendung (wie das Machine Learning Modell) performant zu gestalten, da diese Berechnungen über die API auf einem Server durchgeführt werden müssten. Dies könnte zu Latenzproblemen führen und die Benutzererfahrung negativ beeinflussen.
- Aufgrund der dynamischen Typisierung und Interpretation von JavaScript können Performance- und Typsicherheitsprobleme auftreten, insbesondere in größeren Codebasen.
