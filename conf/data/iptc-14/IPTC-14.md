# IPTC-14 Dataset

The dataset was compiled from the Slovenian news monitoring agency's archive using a predefined set of fourteen IPTC topic tags and a curated list of media outlets.
The selected timespan covers eight months, from 1. 10. 2025 to 1. 6. 2026.

The dataset is available in both JSON lines and CSV formats for convenience.

We first filter articles by creation date and selected media outlets.
Next, we keep only articles that contain at least one of the selected IPTC tags.
Finally, we map the internal tag UUIDs to a normalised label set and to the corresponding IPTC subject codes.

The classification signal comes directly from article tags stored in the archive.
An article can therefore belong to multiple IPTC categories at once.

note: Treat IPTC tags only as an informative classification signal rather than a ground truth.

## Selected IPTC categories

- banking (IPTC 20000274)
- retail (IPTC 20000252)
- insurance (IPTC 20000279)
- culture (IPTC 20000038)
- logistics (IPTC 20000337)
- energy and resource (IPTC 20000256)
- education (IPTC 05000000)
- telecommunication service (IPTC 20000233)
- sport (IPTC 15000000)
- automotive (IPTC 20000296)
- healthcare industry (IPTC 20001354)
- construction and property (IPTC 20000235)
- travel and tourism (IPTC 20000563)
- politics and government (IPTC 11000000)

## Selected media outlets

### National and general web media

- Rtvslo.si (29213ab4-199c-4b11-aa64-eaa37092adf6)
- Siol.net (d53a5e20-a6dd-4ca5-b989-2b662b028f7b)
- Svet24.si (396b201f-2a65-44c8-857e-16c2bf79e55f)
- N1info.si (6a996b7c-b8ea-4aa9-ab67-d8d558d1cba4)
- 24ur.com (1b64e062-3e83-4591-af86-a6e244c45ed5)
- Zurnal24.si (754da261-9aee-4a1a-b9d8-734cd409fabf)
- Metropolitan.si (0a8a2b48-c8cd-4776-9aa5-210fcb999315)
- Nova24tv.si (9aedb9fd-2914-4d8e-9d63-1dbedd03cb65)

### Regional and local web media

- Mojaobcina.si (4e12bd1d-1e5a-4b03-bb45-22ab583905d6)
- Regionalobala.si (b617a5eb-6040-4464-b79f-fba6ddc1bdda)
- Ljubljanainfo.com (2a88961f-67c3-45a8-8527-44196a0c07d2)
- Radio.brezice.eu (f4328a12-99ca-4205-ba96-547a0591f06d)
- Maribor24.si (47562b7d-64e1-480e-8a9d-ccced5c7b31c)
- Regionalgoriska.si (2046495f-8414-4694-999b-4149126d074f)
- Sobotainfo.com (961c0bdc-4c3d-4912-8190-1813ea10295d)
- Dolenjskilist.si (4efb7ff6-78e6-4408-879e-59b9f092a8c9)
- Ptujinfo.com (f00f3e5f-84e8-4a80-83ef-fa342889dd80)
- Mariborinfo.com (82056e16-1268-4b10-a792-ea9bc0b2ef04)
- Dolenjskainfo.com (5e8a68d6-d4dd-4146-90d5-bcca8a2d0322)
- Gorenjskainfo.com (769530df-2ab1-4fae-a8d5-7d693120b7c8)
- Obalaplus.si (88fc87c0-ea40-4a8e-8131-da53de815584)
- Megafon.si (ccfbf5a2-fad0-4f25-a052-ba57ca79a5e3)
- Info07.si (30d2a145-49b6-4d7e-a4f7-61b77455d6cb)
- Info02.si (14dd29b2-0ef4-4dcd-98a1-05cf0bd94719)
- Info03.si (07262734-cc8f-4fc3-bca9-bed587eacb05)
- Info04.si (6fa1ecb9-5fa6-48d8-b8d8-b95fdf1b3ddf)
- Info01.si (69f5c9b7-882b-40dd-9064-17a906ebe832)
- Info05.si (44ae70fa-b64e-498a-b4da-4c6a220ba53b)
- Radiokrka.si (70d86165-2d2f-44c8-bc45-99daaf56ca2b)
- Celje.info (93045532-f197-4da9-9de0-be6795998d7e)

### Specialized media

- Planetnogomet.si (e2bdb7f0-43dc-465d-98fb-03775d5a90b1)
- Avtomanija.com (08552149-f531-451b-8252-317fdc07343e)
- Citymagazine.si (34bb941c-ef75-4975-9b55-f1009b670e35)
- Avto-magazin.si (a646a314-3c3a-48c6-819e-a8aea1c141be)
- Racunalniske-novice.com (83a94c04-7767-496b-a151-3f1b46f4d2c5)
- Stadion.si (e2036e65-7d01-450b-bdbd-ab8216c3593c)
- Avto-fokus.si (2ca990f5-6444-4786-91a5-431448f71d48)
- Caranduser.com (40f5b598-369e-42ad-86a0-585a7fddcb42)
- Automobil.si (ea019ea9-920c-464d-8a93-b7b1f4056128)
- Moskisvet.com (be2fe4a2-8ad9-4434-a615-b5879fc56d43)
- Avtomobilizem.com (3f07f608-472a-45f2-bf2a-3e3e7f588129)
- Avto.info (c11136d2-7cc4-4e31-b636-517b4925c3ba)

## Article structure

Each news article has the following fields:

```json
{
  "uuid": "29007f80-f7ea-11de-a934-02000073f003",
  "created": "2025-10-02T08:15:10.000Z",
  "published": "2025-10-02T07:50:00.000Z",
  "iptc_tags": {
    "20000274": "banking",
    "20000252": "retail"
  },
  "media_uuid": "29213ab4-199c-4b11-aa64-eaa37092adf6",
  "media_name": "Rtvslo.si",
  "section_uuid": "350532d8-d587-4dd7-8626-0d8666d7d7ab",
  "section_name": "Gospodarstvo",
  "media_type": "internet",
  "country": "Slovenia",
  "language": "sl",
  "title": "Banka širi ponudbo digitalnih storitev za trgovce",
  "url": "https://www.rtvslo.si/gospodarstvo/banka-siri-ponudbo-digitalnih-storitev-za-trgovce/123456",
  "body": "Banka je predstavila novo storitev za trgovce, ki povezuje plačilne rešitve, zvestobne programe in analitiko prodaje."
}
```

Notes:

- `iptc_tags` is a mapping from IPTC subject code to the normalized label used in this dataset.
- Articles can contain more than one selected IPTC category.
- Articles without `tags`, `translations`, `language`, `media`, or `country` are skipped during dataset creation.
- The text is taken from `translations[language]`, the article URL is stripped from the body, and duplicated title prefixes are removed.
- The language and country are derived from outlet metadata and can be wrong.
