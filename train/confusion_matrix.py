import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

labels=[0,1,2]
cm = confusion_matrix(y, y_pred, labels=labels)
print(cm)

class_names = [encode_dict[i] for i in labels]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix - Financial News Sentiment Classifier")
# plt.show()
plt.savefig("confusion_matrix.svg", format="svg")
