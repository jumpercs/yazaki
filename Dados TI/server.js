import express from 'express';
import sqlite3 from 'sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const app = express();
const PORT = 3000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.static(path.join(__dirname, 'public')));
app.use('/fotos', express.static(path.join(__dirname, 'fotos')));

let db = new sqlite3.Database('./Dados_ti Banco Pronto.db', (err) => {
  if (err) {
    console.error(err.message);
  }
  console.log('Connected to the database.');
});

app.get('/faces', (req, res) => {
  db.all("SELECT id_rede, nome FROM dados_ti", [], (err, rows) => {
    if (err) {
      throw err;
    }
    res.json(rows);
  });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
