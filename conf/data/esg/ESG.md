# ESD Dataset

The dataset was compiled using a curated list of companies and media outlets from the Slovenian news monitoring agency’s archive.  
The timespan selected was fifteen years from 1. 1. 2010 to 1. 1. 2025.

The dataset is available in both JSON lines and CSV formats for convenience.

We first run a case-insensitive search for articles using keywords and expressions, as the index doesn’t support case sensitivity.  
Next, we matched all keywords and expressions with the result text in a case-sensitive manner, but only where the keywords or expressions were in mixed case.  
We kept only matched articles.

Keywords are used for a simple single-term search. Prefix search was used where inflexions are present (usually non-acronyms).  
For multi-word search, we use span-near-prefix search with distance between terms 0 (i.e. phrase of prefixed terms effectively)

An example best explains the approach used:  
  
We search for Petrol\* with a term prefix search to cover all inflexions (Petrola, Petrolu ...)   
We get back “bla bla bla petrolejska nafta bla bla bla” which is some sort of crude oil fuel.  
So we match against “Petrol\*” and “PETROL\*” from the configuration, but this time we match case-sensitive, so this ("petrolejska nafta") sample is excluded as a false positive.

We used the following configuration:

```yaml
    industry_match:
      talum:
        uuid: ec4c9172-b612-3e0e-809b-72dbfbfb23f3
        keywords:
          - talum*
      sdh:
        uuid: f5f73321-fe7e-34b4-b701-6b41a999f512
        keywords:
          - SDH
        expressions:
          - slovensk* držav* holding*
      zns:
        uuid: 1c4a5d3e-c0f1-3b6a-9b98-610c13505f0a
          - združenj* nadzornik* sloveni*
      nlb:
        uuid: 648d1735-61b5-3f67-b9f5-413596acfe25
        keywords:
          - NLB
        expression:
          - nov* ljubjansk* bank*
      salonit:
        uuid: 74a43247-0343-3a8d-9d5c-b046d90308c1
        expressions:
          - salonit* anhov*
          - alpacem*
      sž:
        uuid: c5dd0ca4-4951-3cd3-915a-3de7993042f7
        expressions:
          - slovensk* železnic*
        keywords:
          - SŽ
      sij:
        uuid: 9612f51c-947b-38b4-98cb-bb06c9a9c07d
        keywords:
          - SIJ
        expressions:
          - slovensk* industrij* jekla
      triglav:
        uuid: 0f7bce51-1fd4-3bc7-b1bb-210442570e2f
        expressions:
          - zavarovaln* triglav*
          - triglav* zavarovaln*
      telekom:
        uuid: 829826a3-0221-3b41-8364-102835c4dfb7
        expressions:
          - telekom* sloven*
      cinkarna:
        uuid: 6318f134-93e2-30d2-b047-63c61e1471e7
        expressions:
          - cinkarn* celje*
      steklarna:
        uuid: 918ceb8d-ff6e-4a3c-a5f3-e2d1ff61f990  # synthetic
        expressions:
          - steklarn* hrastnik*
      sava:
        uuid: 918ceb8d-ff6e-4a3c-a5f3-e2d1ff61f991  # synthetic
        expressions:
          - zavarovaln* sav*
      otp:
        uuid: 918ceb8d-ff6e-4a3c-a5f3-e2d1ff61f992  # synthetic
        expressions:
          - otp* bank*
        keywords:
          - OTP
      pošta:
        uuid: 918ceb8d-ff6e-4a3c-a5f3-e2d1ff61f993  # synthetic
        expressions:
          - pošt* sloven*
      petrol:
        uuid: 918ceb8d-ff6e-4a3c-a5f3-e2d1ff61f994  # synthetic
        keywords:
          - Petrol*
          - PETROL*
      trbovlje:
        uuid: 918ceb8d-ff6e-4a3c-a5f3-e2d1ff61f995  # synthetic
        keywords:
          - RTH
        expressions:
          - rudnik* trbovlj* hrastnik*
          - rudnik* trbovlj*
      šoštanj:
        uuid: 918ceb8d-ff6e-4a3c-a5f3-e2d1ff61f996  # synthetic
        keywords:
          - TEŠ
        expressions:
          - termoelektrarn* šoštanj*
      sid:
        uuid: 918ceb8d-ff6e-4a3c-a5f3-e2d1ff61f997  # synthetic
        keywords:
          - SID
        expressions:
          - sid* bank*
      intereuropa:
        uuid: 918ceb8d-ff6e-4a3c-a5f3-e2d1ff61f998  # synthetic
        keywords:
          - intereurop*
```

## Selected companies

Note: Not all requested companies had media monitoring:

- Talum (ec4c9172-b612-3e0e-809b-72dbfbfb23f3)
- SDH - Slovenski državni holding (f5f73321-fe7e-34b4-b701-6b41a999f512)
- Združenje nadzornikov Slovenije (1c4a5d3e-c0f1-3b6a-9b98-610c13505f0a)
- NLB (648d1735-61b5-3f67-b9f5-413596acfe25)
- Salonit Anhovo (Renamed to Alpacem) (74a43247-0343-3a8d-9d5c-b046d90308c1)
- Slovenske železnice (c5dd0ca4-4951-3cd3-915a-3de7993042f7)
- SIJ - Slovenska industrija jekla (9612f51c-947b-38b4-98cb-bb06c9a9c07d)
- Zavarovalnica Triglav (0f7bce51-1fd4-3bc7-b1bb-210442570e2f)
- Telekom Slovenije (829826a3-0221-3b41-8364-102835c4dfb7)
- Cinkarna Celje (6318f134-93e2-30d2-b047-63c61e1471e7)

The missing companies from the monitoring are:

- Steklarna Hrastnik
- Zavarovalnica SAVA
- OTP Banka
- Pošta Slovenije
- Petrol
- Rudnik Trbovlje-Hrastnik
- TEŠ Termoelektrarna Šoštanj
- Petrol
- SID banka
- Intereuropa

## Selected media outlets

Here’s the rewritten markdown with each title followed by its UUID in parentheses :

### Press

- SLOVENSKE NOVICE (cf08eb8c-1e50-4884-ae0a-9ccf7d735817)
- DELO (2fd717ed-78ba-4f63-b257-cd096acb6bda)
- VEČER (f0f8c65f-3cf8-47e5-a958-5e5372450046)
- DNEVNIK (64b8809a-4143-4609-9a59-eeed8bc180d7)
- NOVICE SVET24 (d09bd49a-e71d-45ff-a30c-c7795d079cda)
- FINANCE (Not available for processing)

### Weekly Press

- NEDELJSKI DNEVNIK (17c2f863-f348-4fe0-9ea5-02e4bedca568)
- MLADINA (2f4bae21-9e4b-4a5f-9168-c1eea24c86c0)
- ONA (5b7318c1-1a7d-4219-8427-fcd730a748eb)
- VIKEND (3456bb2d-4c31-401f-9e3d-d05dd13ae72b)
- NEDELO (7529f6b5-2a31-4359-86e6-3817136a3bb7)
- SOBOTNA PRILOGA (7eadd15d-28aa-485c-8c00-1e11cff87069)

### Web media

- 24ur.com (1b64e062-3e83-4591-af86-a6e244c45ed5)
- zurnal24.si (754da261-9aee-4a1a-b9d8-734cd409fabf)
- siol.net (d53a5e20-a6dd-4ca5-b989-2b662b028f7b)
- metropolitan.si (0a8a2b48-c8cd-4776-9aa5-210fcb999315)
- svet24.si (1d51235d-5f06-4801-bfd2-b549cf788978)
- n1info.si (6a996b7c-b8ea-4aa9-ab67-d8d558d1cba4)
- rtvslo.si (29213ab4-199c-4b11-aa64-eaa37092adf6)

## Article structure

Each news article has the following fields:

```json
{
  "uuid": "29007f80-f7ea-11de-a934-02000073f003",
  "created": "2010-01-02T21:59:55.000Z",
  "published": "2010-01-01T13:46:00.000Z",
  "media_uuid": "29213ab4-199c-4b11-aa64-eaa37092adf6",
  "media_name": "Rtvslo.si",
  "section_uuid": "350532d8-d587-4dd7-8626-0d8666d7d7ab",
  "section_name": "Gospodarstvo",
  "media_type": "internet",
  "country": "SI",
  "language": "sl",
  "title": "Novo leto prinaša višji račun za elektriko",
  "url": "http://www.rtvslo.si/gospodarstvo/novo-leto-prinasa-visji-racun-za-elektriko/220363",
  "body": "... ki bo izbran na javnem razpisu zaradi uporabe domačega premoga iz rudnika Trbovlje Hrastnik (RTH). Vlada je določila, da se v letno energetsko bilanco za leto 2010 uvrsti 391.500 ton rjavega premoga RTH.\r\n\r\nTako bo s sredstvi prispevka za zagotavljanje zanesljive oskrbe z elektriko zagotovljeno, ...",
  "monitored": [ /* monitored client */
    "talum"
  ],
  "monitored_uuid": [
    "ec4c9172-b612-3e0e-809b-72dbfbfb23f3"
  ],
  "matched": [
    "trbovlje" /* see configuration for categorical name */
  ],
  "matches": [ /* see configuration for matching */
    "RTH",
    "rudnik* trbovlj* hrastnik*",
    "rudnik* trbovlj*"
  ]
}
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