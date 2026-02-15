# Modelowanie ogrzewania: Analiza efektywności energetycznej

Projekt realizuje symulację numeryczną rozkładu temperatury w mieszkaniu, mającą na celu odpowiedź na pytanie: **czy warto wyłączać grzejniki podczas nieobecności domowników (8:00–16:00)?**

## Cel projektu
Porównanie dwóch strategii sterowania ogrzewaniem w mieszkaniu o powierzchni 80 m²:
1. **Strategia ciągła:** Stała temperatura zadana 21 stopni Celsjusza przez całą dobę.
2. **Strategia oszczędna:** Obniżenie temperatury w godzinach 8:00–16:00 (nieobecność), powrót do wyjściowej temperatury po południu.

## Metodologia
Symulacja opiera się na rozwiązaniu **dwuwymiarowego równania dyfuzji ciepła z członem źródłowym** (grzejniki):
$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u + f(x,u)$$
Zastosowano metodę różnic skończonych (FDM) oraz jawny schemat całkowania w czasie (FTCS) na siatce $50 \times 40$. Model uwzględnia straty ciepła przez ściany zewnętrzne i okna oraz zmienne warunki pogodowe.

W projekcie przyjęłam następujące kryteria zimna:
- bardzo zimno: od ok. -18 do -8 stopni
- zimno: od ok. -7 do 2 stopni
- chłodno: od ok. +2 do 7 stopni
(oczywiście Celsjusza)

## Kluczowe wyniki
Przeprowadzono symulacje dla trzech scenariuszy pogodowych (od chłodnej jesieni po mroźną zimę).
- **Oszczędność energii:** Strategia oszczędna pozwala zredukować dobowe zużycie energii o **ok. 6%** (niezależnie od temperatury na zewnątrz).
- **Komfort cieplny:** System grzewczy jest na tyle wydajny, że powrót do temperatury komfortowej ($21^\circ$C) po godzinie 16:00 zajmuje zaledwie **10–25 minut**, nawet przy silnych mrozach.

##  Technologie
* **Python 3** (symulacja)
* **NumPy** (algebra liniowa)
* **Matplotlib** (wizualizacje)
