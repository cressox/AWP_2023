## Meilenstein 2 Statusbericht

### Einleitung

Dieser Statusbericht fasst den aktuellen Stand und die Fortschritte im zweiten Meilenstein des Deployments zusammen. Ziel des Meilensteins ist es, die Integration von Mediapipe und OpenCV in das Kivy-Framework zu untersuchen und die App erfolgreich als apk-Datei bereitzustellen. Dabei wurden auch Herausforderungen im Zusammenhang mit dem Kamerazugriff unter Windows adressiert und alternative Lösungsansätze evaluiert.

### Aktueller Stand

Der aktuelle Stand ist wie folgt:

1. **Integration von Kivy und Dlib**: Es wurde erfolgreich Dlib in Kivy integriert, wodurch es möglich ist, den Code auszuführen und die GUI anzuzeigen. Allerdings besteht noch das Problem, dass das Bereitstellen der App als apk-Datei nicht reibungslos funktioniert.

2. **Integration von Mediapipe und OpenCV**: Zur Lösung der Bereitstellungsprobleme wurde die Integration von Mediapipe und OpenCV in das Kivy-Framework untersucht. Dabei wurde das Framework "Mediapipe for Android" verwendet, um eine apk-Datei zu erstellen. Es besteht jedoch noch die Herausforderung, Mediapipe und OpenCV erfolgreich in Kivy zu integrieren, um die gewünschte Funktionalität zu erreichen.

3. **Nutzung eines Docker-Containers**: Um die Erstellung von apk-Dateien mit Mediapipe for Android zu ermöglichen, wurde ein Docker-Container eingerichtet, der die erforderlichen Abhängigkeiten enthält.

4. **Herausforderungen und Lösungsansätze**: Bei der Integration von Mediapipe und OpenCV in Kivy sind noch Herausforderungen zu bewältigen. Es wird intensiv nach Lösungsansätzen gesucht, um die App erfolgreich zum Laufen zu bringen.

5. **Kamerazugriff unter Windows**: Unter Windows besteht die Schwierigkeit, dass der Kamerazugriff zwischen Mediapipe und Kivy nicht reibungslos funktioniert. Aus diesem Grund wurde beschlossen, ein natives Linux-System auf einem Laptop zu verwenden, um die App unter optimalen Bedingungen zu testen.

### Nächste Schritte

Die nächsten Schritte sind wie folgt geplant:

1. **Behebung von Integrationsproblemen**: Es wird weiterhin intensiv daran gearbeitet, die Integrationsprobleme zwischen Mediapipe, OpenCV und Kivy zu lösen. Dazu werden verschiedene Ansätze und Lösungen untersucht, um die Funktionalität der App sicherzustellen.

2. **Testen unter Linux**: Um den Kamerazugriff problemlos zu ermöglichen, wird ein natives Linux-System auf einem Laptop verwendet. Dadurch soll eine umfassende Testumgebung geschaffen werden, um die Landmarkenerkennung und die gesamte App-Funktionalität erfolgreich zu überprüfen.

3. **Weiterentwicklung und Optimierung**: Sobald die Integration von Mediapipe und OpenCV in Kivy erfolgreich abgeschlossen ist, wird die weitere Entwicklung der App vorangetrieben.

4. **Bereitstellung als apk-Datei**: Das langfristige Ziel besteht darin, die App erfolgreich als apk-Datei bereitzustellen, um sie auf Android-Geräten nutzen zu können.

zuzu