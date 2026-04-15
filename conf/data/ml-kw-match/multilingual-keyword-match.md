# Multilingual Keyword Matched Dataset

The dataset was compiled using a curated list of single- and multiword phrases matched against text. The source material originates from the Slovenian news monitoring agency’s archive and comprises news and social media texts in Bosnian, Croatian, Macedonian, Serbian, and Slovenian. Additional texts from other countries, such as the Czech Republic, Poland, Russia, Romania, Bulgaria, and Ukraine, may also be present. The selected timespan covers twenty years, from January 1, 2005, to January 1, 2025.

The dataset is available in both JSON lines and CSV formats for convenience.

## Selected keywords

### The United States

- USA
- U.S.A.
- United States
- United States of America
- ZDA
- ZDA-ja
- ZDA-ju
- SAD
- SAD-u
- SAD-a
- SAD-om
- SAD-e
- САД
- САД-а
- САД-у
- САД-е
- САД-ом
- Združene Države Amerike
- Združenih Držav Amerike
- Sjedinjene Američke Države
- Sjedinjenih Američkih Država
- Сједињене Америчке Државе
- Соединети Американски Држави
- Соединетите Американски Држави

### The North Atlantic Treaty Organization

- NATO-m
- NATO
- NATA
- NATU
- NATOM
- NATE
- НАТО
- НАТА
- НАТУ
- НАТОМ
- НАТО-m
- НАТЕ
- North Atlantic Treaty Organization
- Severnoatlantska zveza
- Severnoatlantske zveze
- Severnoatlantski savez
- Severnoatlantskog saveza
- Severnoatlantskom savezu
- Северноатлантски савез
- Северноатлантског савеза
- Северноатлантском савезу
- Северно атлантски сојуз
- Северно атлантскиот сојуз

### Tesla - Automotive Company

- Tesla
- TESLA
- Tesle
- TESLE
- Tesli
- TESLI
- Teslo
- TESLO
- Teslu
- TESLU
- Teslom
- TESLOM
- Тесла
- ТЕСЛА
- Тесле
- ТЕСЛЕ
- Тесли
- ТЕСЛИ
- Тесло
- ТЕСЛО
- Теслу
- ТЕСЛУ
- Теслом
- ТЕСЛОМ
- Тесла
- ТЕСЛА
- Тесли
- ТЕСЛИ
- Тесла
- Тесљa
- ТЕСЉА
- Теслата
- ТЕСЛАТА
- Теслава
- ТЕСЛАВА
- Теслана
- ТЕСЛАНА

### IKEA - The furniture company

- Ikea
- IKEa
- Ikee
- IKEE
- Ikei
- IKEI
- Ikeo
- IKEO
- Ikeu
- IKEU
- Ikeom
- IKEOM
- Икеа
- ИКЕА
- Икее
- ИКЕЕ
- Икеи
- ИКЕИ
- Икео
- ИКЕО
- Икеу
- ИКЕУ
- Икеом
- ИКЕОМ
- Икеја
- ИКЕЈА
- Икеата
- ИКЕАТА
- Икеава
- ИКЕАВА
- Икеана
- ИКЕАНА
- Икејa
- ИКЕЈА

### Volkswagen - Automotive Company

- Volkswagen
- Volkswagna
- Volkswagnu
- Volkswagnom
- Volkswagni
- Volkswagnih
- Volkswagne
- VW
- VWa
- VWu
- VWom
- VWi
- VWh
- VWe
- Volkswagena
- Volkswagenu
- Volkswagenom
- Volkswagene
- Volkswagenu
- Фолксваген
- Фолксвагена
- Фолксвагену
- Фолксвагеном
- Фолксвагене
- Фолксвагену
- Фолксвагенот
- Фолксвагени

### Global Warming - General Concept

- globalno segrevanje
- globalnega segrevanja
- globalnemu segrevanju
- globalnem segrevanju
- globalnim segrevanjem
- globalno zagrijavanje
- globalnog zagrijavanja
- globalnom zagrijavanju
- globalnim zagrijavanjem
- globalno zatopljenje
- globalnog zatopljenja
- globalno zagrevanje
- globalnog zagrevanja
- globalnom zagrevanju
- globalnim zagrevanjem
- globalnom zatopljenju
- globalnim zatopljenjem
- глобално загревање
- глобалног загревања
- глобалном загревању
- глобалним загревањем
- глобално затопљење
- глобалног затопљења
- глобалном затопљењу
- глобалним затопљењем
- глобално затоплување
- глобалното затоплување

## Article structure

Each news article has the following fields:

```json
{
  "uuid": "b2cf7800-bb3a-11db-bbb1-020000b24cbb",
  "created": "2007-02-13T08:17:52.000Z",
  "published": "2007-02-12T23:00:00.000Z",
  "categories": [
    "zda"
  ],
  "media_uuid": "d92409bd-2b39-4aac-8782-4cd0ee3e278a",
  "media_name": "Poslovni dnevnik",
  "section_uuid": "0c775644-f9b6-4fd8-bdd8-d297fd51b209",
  "section_name": "Ostalo",
  "media_type": "print",
  "country": "HR",
  "language": "hr",
  "title": "Nectar preradio više od 100.000 tona voća",
  "url": "",
  "body": "Nectar Je 2003. u pogon za proizvodnju koncentrata i kaša investirao dva milijuna eura NOVI SAD U pogonima kompanije Nectar u Bačkoj Palanci lani je preradeno više od 100.000 tona raznog voča, a proizvodi od njega plasirani su od Hong Konga preko Australije do SAD ..."
}
```

The categories are:

```
zda
nato
tesla
ikea
volkswagen
global_warming
```

Notes:

- The language and country are derived from an outlet and can be wrong.
- Press media can have OCR image artefacts in the text body:

  ```
  težave z gmotnim položajem in revščino. _^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_mt- 
  J________________H '^1E H— 3SpP^^™\"^^^H^^^^^^^^^^^^^^^^^^^^^^^^^H - 
  - ®fl W_E. i Jg|^^^^^^^^^^^^^^^| MMiH— — 5 5BIIBI_B—_^__^__^__^__^__^__^__^__^__i 
  Višino mesečnih dohodkov upokojencev v perspektivo
  ```