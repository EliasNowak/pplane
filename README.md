# PPlane Modern

Moderne WebApp zur Visualisierung von Phasenporträts für 2D-Differentialgleichungen.

## Features

- Richtungsfeld (abschaltbar)
- Nullklinen-Overlay (dx/dt = 0 und dy/dt = 0)
- Automatische Erkennung von Gleichgewichtspunkten (approx.)
- Interaktive Trajektorien per Klick in den Plot
- Eingabefelder für eigene Gleichungen `dx/dt`, `dy/dt`
- Parameter `a, b, c, d`
- Vorgefertigte Presets (u. a. Lotka-Volterra, Van der Pol, Pendel)

## Entwicklung starten

```bash
npm install
npm run dev
```

Danach unter `http://localhost:3000` öffnen.

## Produktion

```bash
npm run build
npm start
```

## Hinweise

- Parser basiert auf `mathjs`
- Unterstützte Variablen: `x, y, a, b, c, d`
- Unterstützte Funktionen: `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, etc.
