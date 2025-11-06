"""
Example: Kreditkarten-Betrugserkennung mit Gradient Boosting
=============================================================
Dieses Beispiel simuliert ein realistisches Szenario aus dem Bankwesen:
Die Erkennung von betrügerischen Kreditkartentransaktionen.

Szenario:
---------
Eine Bank möchte verdächtige Kreditkartentransaktionen automatisch erkennen.
Dabei werden verschiedene Merkmale der Transaktion analysiert:
- Transaktionsbetrag (in Euro)
- Abstand zum durchschnittlichen Transaktionsbetrag des Kunden
- Zeitpunkt der Transaktion (ungewöhnliche Uhrzeiten)
- Geografische Entfernung zur letzten Transaktion
- Anzahl der Transaktionen in den letzten 24 Stunden

Das Modell lernt aus historischen Daten, welche Kombinationen dieser Merkmale
auf betrügerische Aktivitäten hindeuten.
"""
import numpy as np
from gradient_boosting import GradientBoostingClassifier
from visualization import GradientBoostingVisualizer, PerformanceVisualizer


def generate_credit_card_data(n_samples=400):
    """
    Generiert realistische Kreditkarten-Transaktionsdaten.

    Features:
    - Feature 0: Transaktionsbetrag (normalisiert, 0-1000€)
    - Feature 1: Zeitpunkt (0-24 Uhr, normalisiert)
    - Feature 2: Geografische Entfernung zur letzten Transaktion (km, normalisiert)
    - Feature 3: Anzahl Transaktionen in letzten 24h
    - Feature 4: Abweichung vom durchschnittlichen Transaktionsbetrag

    Returns:
    - X: Feature-Matrix (n_samples, 5)
    - y: Labels (0 = legitim, 1 = betrügerisch)
    - feature_names: Namen der Features
    - transactions_info: Liste mit detaillierten Transaktionsbeschreibungen
    """
    np.random.seed(42)

    # Normale (legitime) Transaktionen
    n_normal = int(n_samples * 0.85)  # 85% legitime Transaktionen
    n_fraud = n_samples - n_normal    # 15% Betrug

    # Legitime Transaktionen:
    # - Kleinere Beträge (10-200€)
    # - Normale Uhrzeiten (8-22 Uhr)
    # - Kleine geografische Entfernungen
    # - Wenige Transaktionen pro Tag (1-5)
    # - Geringe Abweichung vom Durchschnitt
    normal_amount = np.random.normal(80, 40, n_normal).clip(10, 200) / 1000
    normal_time = np.random.normal(15, 4, n_normal).clip(8, 22) / 24
    normal_distance = np.random.exponential(5, n_normal).clip(0, 50) / 100
    normal_count = np.random.poisson(2, n_normal).clip(1, 5) / 10
    normal_deviation = np.random.normal(0, 0.1, n_normal).clip(-0.3, 0.3)

    X_normal = np.column_stack([
        normal_amount, normal_time, normal_distance, normal_count, normal_deviation
    ])

    # Betrügerische Transaktionen:
    # - Höhere Beträge (200-800€)
    # - Ungewöhnliche Uhrzeiten (nachts: 0-6 Uhr oder sehr spät)
    # - Große geografische Entfernungen (Ausland)
    # - Viele Transaktionen kurz hintereinander
    # - Hohe Abweichung vom normalen Verhalten
    fraud_amount = np.random.normal(450, 150, n_fraud).clip(200, 800) / 1000
    # Nachts oder sehr früh morgens
    fraud_time_choice = np.random.choice([0, 1], n_fraud)
    fraud_time = np.where(
        fraud_time_choice,
        np.random.uniform(0, 6, n_fraud),  # Nachts
        np.random.uniform(22, 24, n_fraud)  # Spät abends
    ) / 24
    fraud_distance = np.random.exponential(200, n_fraud).clip(50, 1000) / 100
    fraud_count = np.random.poisson(8, n_fraud).clip(5, 20) / 10
    fraud_deviation = np.random.normal(0.5, 0.2, n_fraud).clip(0.3, 1.0)

    X_fraud = np.column_stack([
        fraud_amount, fraud_time, fraud_distance, fraud_count, fraud_deviation
    ])

    # Kombiniere die Daten
    X = np.vstack([X_normal, X_fraud])
    y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])

    # Mische die Daten
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    feature_names = [
        'Betrag (€)',
        'Uhrzeit',
        'Entfernung (km)',
        'Anz. Trans./24h',
        'Abweichung ø'
    ]

    # Erstelle Beispiel-Transaktionsbeschreibungen für die ersten 5 Transaktionen
    transactions_info = []
    for i in range(min(5, n_samples)):
        amount = X[i, 0] * 1000
        time = X[i, 1] * 24
        distance = X[i, 2] * 100
        count = X[i, 3] * 10
        fraud_label = "[BETRUG]" if y[i] == 1 else "[Legitim]"

        info = f"""
    Transaktion #{i+1}: {fraud_label}
    - Betrag: {amount:.2f}€
    - Uhrzeit: {int(time):02d}:{int((time % 1) * 60):02d} Uhr
    - Entfernung zur letzten Transaktion: {distance:.1f} km
    - Transaktionen in 24h: {int(count)}
    """
        transactions_info.append(info)

    return X, y, feature_names, transactions_info


def main():
    print("=" * 80)
    print("  KREDITKARTEN-BETRUGSERKENNUNG MIT GRADIENT BOOSTING")
    print("=" * 80)
    print("\n Szenario:")
    print("   Eine Bank analysiert Kreditkartentransaktionen, um Betrug zu erkennen.")
    print("   Das System lernt aus historischen Daten, verdächtige Muster zu identifizieren.")
    print("=" * 80)

    # Initialize output handler for this run
    from visualization import OutputHandler
    session_dir = OutputHandler.initialize_session()
    print(f"\n Visualisierungen werden gespeichert in: {session_dir}")

    # Generate data
    print("\n1. DATENGENERIERUNG")
    print("   " + "-" * 70)
    X, y, feature_names, transactions_info = generate_credit_card_data(n_samples=400)

    print(f"   Anzahl Transaktionen: {len(X)}")
    print(f"   Legitime Transaktionen: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"   Betrügerische Transaktionen: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")

    print("\n   Beispiel-Transaktionen:")
    for trans_info in transactions_info:
        print(trans_info)

    # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\n   Trainingsdaten: {len(X_train)} Transaktionen")
    print(f"   Testdaten: {len(X_test)} Transaktionen")

    # Train model
    print("\n2. MODELL-TRAINING")
    print("   " + "-" * 70)
    print("   Trainiere Gradient Boosting Klassifikator...")
    print(f"   Konfiguration:")
    print(f"   - Anzahl Bäume: 50")
    print(f"   - Lernrate: 0.1")
    print(f"   - Maximale Tiefe: 3")
    print(f"   - Subsample: 80%")

    model = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        subsample=0.8
    )
    model.fit(X_train, y_train)
    print("   Training abgeschlossen!")

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_train = model.predict_proba(X_train)
    y_proba_test = model.predict_proba(X_test)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_auc = roc_auc_score(y_train, y_proba_train)
    test_auc = roc_auc_score(y_test, y_proba_test)

    # Zusätzliche Metriken für Fraud Detection
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)

    print(f"\n3. MODELL-PERFORMANCE")
    print("   " + "-" * 70)
    print(f"   Trainingsdaten:")
    print(f"      Genauigkeit: {train_acc:.1%}")
    print(f"      AUC-Score: {train_auc:.3f}")
    print(f"\n   Testdaten (neue, ungesehene Transaktionen):")
    print(f"      Genauigkeit: {test_acc:.1%}")
    print(f"      AUC-Score: {test_auc:.3f}")
    print(f"      Präzision: {test_precision:.1%} (Wie viele erkannte Betrugsfälle sind echt?)")
    print(f"      Recall: {test_recall:.1%} (Wie viele echte Betrugsfälle werden erkannt?)")
    print(f"      F1-Score: {test_f1:.3f}")

    # Zeige einige Beispiel-Vorhersagen
    print(f"\n   Beispiel-Vorhersagen auf Testdaten:")
    example_indices = np.random.choice(len(X_test), size=min(5, len(X_test)), replace=False)
    for idx in example_indices:
        y_actual = y_test[idx] if isinstance(y_test, np.ndarray) else y_test.iloc[idx]
        actual = "Betrug" if y_actual == 1 else "Legitim"
        predicted = "Betrug" if y_pred_test[idx] == 1 else "Legitim"
        confidence = y_proba_test[idx] * 100
        correct = "[OK]" if y_actual == y_pred_test[idx] else "[FALSCH]"

        x_sample = X_test[idx] if isinstance(X_test, np.ndarray) else X_test.iloc[idx].values
        amount = x_sample[0] * 1000
        time = x_sample[1] * 24

        print(f"\n      {correct} Transaktion: {amount:.2f}€ um {int(time):02d}:{int((time % 1) * 60):02d} Uhr")
        print(f"        Tatsächlich: {actual} | Vorhergesagt: {predicted} (Konfidenz: {confidence:.1f}%)")

    # Visualizations
    print(f"\n4. VISUALISIERUNGEN")
    print("   " + "-" * 70)

    # Initialize visualizers
    gb_viz = GradientBoostingVisualizer(model)
    perf_viz = PerformanceVisualizer(model, X_train, y_train, X_test, y_test)

    # Plot training loss
    print("   Erstelle Trainingsverlauf...")
    gb_viz.plot_training_loss()

    # Plot feature importance
    print("   Analysiere Feature-Wichtigkeit...")
    print("       (Welche Transaktionsmerkmale sind am wichtigsten für die Betrugserkennung?)")
    gb_viz.plot_feature_importance(feature_names=feature_names)

    # Plot confusion matrix
    print("   Erstelle Confusion Matrix...")
    print("       (Visualisiert korrekte vs. falsche Klassifikationen)")
    perf_viz.plot_confusion_matrix()

    # Plot ROC curve
    print("   Erstelle ROC-Kurve...")
    print("       (Trade-off zwischen True Positive und False Positive Rate)")
    perf_viz.plot_roc_curve()

    # Plot learning curve
    print("   Erstelle Learning Curve...")
    print("       (Zeigt, wie die Performance mit mehr Trainingsdaten steigt)")
    perf_viz.plot_learning_curve()

    # Geschäftliche Interpretation
    print(f"\n5. GESCHÄFTLICHE INTERPRETATION")
    print("   " + "-" * 70)

    # Berechne potenzielle Einsparungen
    fraud_in_test = np.sum(y_test)
    fraud_detected = np.sum((y_test == 1) & (y_pred_test == 1))
    avg_fraud_amount = 450  # Euro (Durchschnitt betrügerischer Transaktionen)

    prevented_loss = fraud_detected * avg_fraud_amount
    missed_fraud = (fraud_in_test - fraud_detected) * avg_fraud_amount

    false_positives = np.sum((y_test == 0) & (y_pred_test == 1))
    inconvenience_cost = false_positives * 5  # 5€ Kosten pro fälschlich blockierter Transaktion

    print(f"   Potenzielle Einsparungen:")
    print(f"      Erkannte Betrugsfälle: {fraud_detected}/{fraud_in_test}")
    print(f"      Verhinderte Verluste: ~{prevented_loss:.0f}€")
    print(f"      Übersehener Betrug: ~{missed_fraud:.0f}€")
    print(f"\n   Falsch-Positive:")
    print(f"      Fälschlich blockierte legitime Transaktionen: {false_positives}")
    print(f"      Geschätzte Unannehmlichkeitskosten: ~{inconvenience_cost:.0f}€")
    print(f"\n   Net-Benefit: ~{prevented_loss - inconvenience_cost:.0f}€")

    print("\n" + "=" * 80)
    print("ANALYSE ABGESCHLOSSEN!")
    print(f"Alle Visualisierungen wurden gespeichert in: {session_dir}")
    print("=" * 80)
    print("\nErkenntnisse:")
    print("   - Das Modell kann verdächtige Transaktionsmuster effektiv erkennen")
    print("   - Wichtigste Indikatoren sind sichtbar in der Feature-Importance-Grafik")
    print("   - Der ROC-AUC-Score zeigt die Gesamtqualität des Modells")
    print("   - In der Praxis würde man den Schwellenwert anpassen, um das")
    print("     Verhältnis von erkanntem Betrug zu Kundenunannehmlichkeiten zu optimieren")
    print("=" * 80)



if __name__ == "__main__":
    main()

