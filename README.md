# Scientific Literature Search Assistant using RAG

## Project Motivation
This project implements a Retrieval Augmented Generation (RAG) system to search scientific papers. As a student, it's always hard to find good matching scientific Papers on Google for research purposes. For this problem i wanted to develop a RAG System, so that it becomes easier to find scientific Papers when doing a reasearch project f.e. the bachelor thesis.

## Getting Started
### Prerequisites
- Python 3.8+
- Virtual Environment / installation using requirements.txt

### Installation
1. Clone the repository:
   ```bash
   git clone alepeco/RAG_Project_ML2
   cd alepeco/RAG_Project_ML2
2. Download the data from https://www.kaggle.com/datasets/Cornell-University/arxiv, place it into the /data/raw folder, and make the data shorter as it takes a lot     of time for embedding. In my case, I used about 24000 records and it took 3.5 hours to create the embeddings
3. Start "src/data_preparation_csv.py". This will place a CSV in the processed folder including the embeddings.
4. Start "src/data_preprocessing.py". This will preprocess the data and create training & validation sets and place them in the folder "data"
5. Start "src/train.py". Here, a Neural Network is trained and validated to create the best refined embeddings to query for the data.
6. Start "App.py" and query for a , the result will be the metadata and abstract of matching scientific papers.

### Validation
![alt text](image-3.png)
(Referenziert Diagramm /validation/images/Figure_Model_training.png)
 
Das Diagramm zeigt die Trainings- und Validierungsverluste über die Epochen während des Hyperparameter-Tuning-Prozesses. Die Trainings- und Validierungsverluste konvergieren im Allgemeinen zu ähnlichen Werten, was darauf hindeutet, dass das Modell lernt und recht gut auf den Validierungssatz verallgemeinert. Periodische Ausschläge in den Trainings- und Validierungsverlusten sind auf die Änderungen der Hyperparameter in den verschiedenen Hyperparameter-Abstimmungsiterationen zurückzuführen. Jeder Satz von Hyperparametern beginnt das Training bei null, was zu einem anfänglich höheren Verlust führt, der mit dem Lernen des Modells abnimmt. Die enge Verfolgung der Trainings- und Validierungsverluste deutet darauf hin, dass das Modell nicht signifikant overfitted ist. Wären die Trainingsverluste viel niedriger als die Validierungsverluste, würde dies auf eine overfitting hindeuten. Die niedrigsten Punkte in der Validierungsverlustkurve stellen die beste Leistung des Modells mit einem bestimmten Satz von Hyperparametern dar. Diese Punkte sind entscheidend für die Identifizierung des besten Modells. Die abrupten Änderungen in den Verlustwerten entsprechen wahrscheinlich Änderungen in den Hyperparametern, was darauf hindeutet, dass einige Hyperparameterkonfigurationen zu einer besseren Anfangsleistung führen als andere.
Aus dem Diagramm geht hervor, dass die Trainings- und Validierungsverluste des Modells mit der Zeit abnehmen und konvergieren, was auf ein effektives Lernen hindeutet. 

![alt text](image-2.png)
(Referenziert Histogramm /validation/images/Figure_cosine_Similarity.png)
 
Average Original Cosine Similarity: 0.7294282913208008
Average Refined Cosine Similarity: 0.8814812898635864
Die bereitgestellten Histogramme und durchschnittlichen Cosinus-Ähnlichkeitswerte geben Aufschluss über die Verfeinerung der Embeddings. Das Histogramm für die ursprünglichen Embeddings zeigt eine Verteilung der Cosinus-Ähnlichkeiten, die um 0,7 zentriert ist, was auf eine mässige Ähnlichkeit zwischen den meisten Paaren von Embeddings hinweist, wobei einige Paare weniger und andere mehr Ähnlichkeit aufweisen. Im Gegensatz dazu liegt das Histogramm für die verfeinerten Embeddings bei 0,88, was auf ein höheres Mass an Ähnlichkeit und eine konsistentere Ähnlichkeitsverteilung zwischen den verfeinerten Embeddings hinweist.
Die durchschnittlichen Cosinus-Ähnlichkeitswerte verdeutlichen diese Verbesserung noch weiter. Die durchschnittliche ursprüngliche Cosinus-Ähnlichkeit beträgt 0,729, was ein mittleres Mass an Ähnlichkeit widerspiegelt. Nach der Verfeinerung steigt die durchschnittliche Cosinus-Ähnlichkeit auf 0,881, was darauf hindeutet, dass die verfeinerten Embeddings einander im Allgemeinen ähnlicher sind. Dieser Anstieg der Ähnlichkeit und die engere Verteilung in den verfeinerten Embeddings deuten darauf hin, dass der Autoencoder-Trainingsprozess die Embeddings erfolgreich in einen besser strukturierten und aussagekräftigeren latenten Raum abbildet.
Insgesamt zeigen die Ergebnisse, dass der Verfeinerungsprozess die Ähnlichkeit und Konsistenz der Embeddings effektiv erhöht hat, wodurch sie sich besser für Aufgaben eignen, die qualitativ hochwertige, semantisch aussagekräftige Repräsentationen erfordern, wie z. B. Information Retrieval und Clustering.

 ![alt text](image.png)
Nearest Neighbors Consistency Score: Der Wert für die Konsistenz der nearest Neighbors von 73,34 % bedeutet, dass etwa 73,34 % der nearest Neighbors in den ursprünglichen embeddings in den verfeinerten embeddings gleich bleiben. Dies deutet darauf hin, dass der Verfeinerungsprozess einen erheblichen Teil der ursprünglichen Nachbarschaftsstruktur bewahrt hat.
Die Bewertung der Konsistenz der nächsten Nachbarn zeigt, dass der Verfeinerungsprozess ein gutes Mass an Konsistenz in der Nachbarschaftsstruktur der Embeddings beibehalten hat. Obwohl es noch Raum für Verbesserungen gibt, zeigt ein Ergebnis von 73,34 % an, dass die verfeinerten Embeddings in Bezug auf die nächsten Nachbarn weitgehend mit den ursprünglichen Embeddings konsistent sind. Diese Konsistenz ist ein positives Ergebnis und deutet darauf hin, dass der Verfeinerungsprozess die lokalen Beziehungen innerhalb des Einbettungsraums nicht wesentlich verändert hat.
Die Trainingsdaten bestanden aus einem vielfältigen Satz von arXiv-Abstracts und den entsprechenden Embeddings. Die Qualität und Quantität dieser Daten waren entscheidend für die Erzielung aussagekräftiger Ergebnisse. Qualitativ hochwertige Embeddings aus dem BERT-Modell boten eine gute Ausgangsbasis für den Autoencoder. Der Datensatz war ausreichend gross, um dem Modell eine gute Generalisierung zu ermöglichen und verschiedene Nuancen in den Daten zu erfassen. Etwaige Verzerrungen oder Einschränkungen im Datensatz würden sich jedoch direkt auf die Leistung des Modells auswirken, was unterstreicht, wie wichtig die Verwendung umfassender und repräsentativer Daten ist.

Validierung:
Die Validierung erfolgte in mehreren Schritten, einschliesslich der Verfolgung der Trainings- und Validierungsverluste, der Kosinus-Ähnlichkeitsanalyse und der Nearest Neighbors Konsistenz. Die Ergebnisse zeigten einen stetigen Rückgang des Trainings- und Validierungsverlustes, was auf einen effektiven Lernprozess des Modells hindeutet. Die durchschnittlichen Cosinus-Ähnlichkeitswerte verbesserten sich von 0,729 auf 0,881, was die Wirksamkeit des Autoencoders bei der Verfeinerung der Embeddings belegt. Der Nearest Neighbors Consistency Score von 73,34 % zeigt, dass die lokale Struktur der Embeddings gut erhalten wurde, obwohl es noch Raum für Verbesserungen gibt. Diese Validierungsschritte bestätigten die Fähigkeit des Modells, die Qualität der Embeddings unter Beibehaltung der ursprünglichen Nachbarschaftsstruktur zu verbessern.
Zusammenfassend lässt sich sagen, dass die bei der Entwicklung des Modells getroffenen Entscheidungen - von der Auswahl einer Autoencoder-Architektur über die Abstimmung der Hyperparameter bis hin zur sorgfältigen Berücksichtigung der Trainingsdaten - alle zur erfolgreichen Verfeinerung der Embeddings beigetragen haben. Die Validierungsergebnisse zeigten signifikante Verbesserungen, die den Ansatz validierten und eine solide Grundlage für weitere Verbesserungen bildeten.
