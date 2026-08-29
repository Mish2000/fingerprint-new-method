# ניסוי 003 — פרוטוקול קפוא לבירור הקשר בין `annotated_512` ל־`final_320`

**סטטוס:** הוקפא לפני ניתוח פיקסלים משמעותי, 2026-08-29T13:06:26Z.

**ענף עבודה:** `experiment/003-l3sf-crosswalk`.

**נקודת בסיס:** `f88929d6bbf6ed453655e07df3869f913b6065eb`.

**תחום:** כל 740 תמונות `annotated_512` וכל 7,400 תמונות `final_320` תחת עץ
`L3_SF_V2/L3SF_V2`, בקריאה בלבד דרך `fingerprint_new_method.paths`. אין שימוש ב־SD300,
אין שינוי ב־`fingerprint-benchmark`, אין אימון מודל, ואין שימוש בקואורדינטות נקבובית לגילוי זהות או
ל־registration.

## 1. שאלות והכרעות

הניסוי יענה, בסדר זה, על שלוש שאלות:

1. מהו התפקיד הנתמך בראיות של `annotated_512` ב־pipeline המקומי של L3-SF?
2. האם אפשר להקפיא crosswalk אמין מ־`annotated_512` לזהות `final_320` באותו run?
3. רק אם שער הזהות עובר: האם affine registration המבוסס על רכסים מאפשר להעביר נקודות נקבובית
   לחלק משמעותי מן ה־final impressions?

עד להכרעה ייעשה שימוש רק במונחים `annotated_512` ו־`final_320`. שלוש ההכרעות האפשריות הן:

- מעבר זהות וגאומטריה (`H2`);
- מעבר זהות בלבד (`H1`);
- אין crosswalk אמין (`H0`).

לאחר ההכרעה הניסוי נעצר. לא יפותחו detector, descriptor או matcher נוספים.

## 2. ביקורת סמנטית, לפני matching

הביקורת תסרוק ללא כתיבה לעץ הנתונים:

- כל README, טקסט, manifest, archive וקובץ metadata;
- מבנה תיקיות ושמות קבצים, לרבות cardinality ויחסי 2×5;
- timestamps ברמת קובץ רק כ־provenance חלש, לא כמפתח correspondence;
- container metadata של PNG/JPEG, לרבות EXIF, ICC, comments ו־text chunks;
- dimensions, modes ופורמטים ללא הסקת משמעות מן הממד בלבד;
- חפיפות SHA-256 ו־byte-identical files בין הענפים;
- נוכחות או היעדר crosswalk מפורש.

`semantic_audit.json` יפריד בין `proved`, ‏`supported`, ‏`plausible` ו־`not_found`.
מספרים משותפים בשמות, סדר directory listing ו־timestamps לא ייחשבו הוכחת correspondence.

## 3. parsing ומרחב מועמדים

- `final_320`: שם חוקי הוא `<identity>_<capture_group>_<instance>.png`, כאשר identity חיובי,
  `capture_group` הוא 1 או 2 ו־`instance` הוא 1..5.
- `annotated_512`: שם חוקי הוא `<local_index>_<pattern>.jpg`; `local_index` הוא מזהה מקומי בתוך
  pattern ולא מזהה `final_320`.
- המזהים הקנוניים הם `R{run}/<stem>` ו־`R{run}/<numeric_identity>` בהתאמה.
- לכל `annotated_512` ייבדקו בדיוק 148 קבוצות הזהות של אותו run. לא יבוצע חיפוש בין runs.
- כל קבוצה תיוצג על ידי כל עשרת ה־impressions שלה. לא ייעשה שימוש במספר שבשם `annotated_512`
  לצורך score, tie-break או assignment.

## 4. מדגמים קפואים ועיוורון

### 4.1 ביקורת crosswalk של 50 יחידות

הזרע הוא `l3sf-exp003-crosswalk-review-v1`. בכל run תיבחר מכסה של ארבע `right_loop`, שלוש
`whorl`, אחת `left_loop`, אחת `plain_arch` ואחת `tented_arch`. בתוך stratum הדירוג הוא
`SHA256(seed|run|canonical_annotated_id)` בסדר הקסדצימלי. המדגם וקובצי המקור ייכנסו ל־manifest
לפני חישוב crosswalk כלשהו וללא החלפה לפי איכות.

לאחר הקפאת המיפוי יופק לכל יחידה לוח מקומי ובו `annotated_512`, שלושה impressions מן המועמד
האוטומטי ושלושה מן המועמד החלופי. בכל קבוצה ייבחרו impressions 1, 6, 10 לאחר מיון קנוני, כדי
לא לבחור את הקלים. שתי הקבוצות יוצגו כ־A/B בסדר שנקבע באמצעות
`SHA256(seed|blind-side|canonical_annotated_id)`, ללא identifiers של identity. הסוקר יבחר
`A`, ‏`B`, ‏`BOTH_UNCERTAIN` או `NEITHER` וייתן confidence `CLEAR` או `AMBIGUOUS` לפני חשיפת
הצד האוטומטי. התרגום יהיה:

- `CLEAR`: בחירה יחידה במועמד האוטומטי עם confidence `CLEAR`;
- `AMBIGUOUS`: בחירה במועמד האוטומטי עם confidence `AMBIGUOUS`, או `BOTH_UNCERTAIN`;
- `NOT_MATCH`: בחירה יחידה בחלופה או `NEITHER`.

### 4.2 מדגם geometry מותנה

רק לאחר מעבר שער 1 והקפאת crosswalk ייבחרו 30 יחידות חזקות, שש מכל run. תינתן עדיפות מכסה
באותו סדר patterns כמו לעיל ככל שהיחידות החזקות מאפשרות, ובכל stratum הדירוג יהיה
`SHA256(l3sf-exp003-geometry-v1|canonical_annotated_id)`. לא תהיה החלפה לפי איכות.

לכל יחידה ייבחר impression אחד מכל capture group. בתוך הקבוצה ייבחר המינימום לפי
`SHA256(l3sf-exp003-impression-v1|canonical_final_sample_id)`, ללא שימוש ב־matching score.
היעד הוא 60 זוגות.

## 5. עיבוד משותף שאינו משתמש בנקבוביות

התמונה מומרת ל־grayscale באמצעות OpenCV, עוברת percentile normalization בין p1 ל־p99
ולאחר מכן CLAHE עם `clipLimit=2.0` ו־tiles בגודל 8×8. אין קריאה של TSV בשלב crosswalk.
כל coordinates מנורמלים ל־[0,1] לצורך חישוב coverage; residual של התאמה גאומטרית נמדד
במערכת `final_320`.

שני הערוצים משתמשים באותו aggregation אך ב־features שונים. לכל impression נשמרים raw score,
מספר correspondences, מספר inliers, inlier ratio, residual ו־coverage בשני הצדדים.

## 6. ערוץ K — keypoints ו־robust affine geometry

1. יופקו לכל תמונה עד 384 keypoints של SIFT עם `contrastThreshold=0.02`, ‏`edgeThreshold=10`
   ו־`sigma=1.6`.
2. descriptor יעבור L1 normalization ולאחר מכן square root (`RootSIFT`).
3. לכל run ייבנה FLANN KD-tree דטרמיניסטי על descriptors של כל `final_320` ב־run
   (`trees=8`, ‏`checks=256`). לכל descriptor של `annotated_512` יוחזרו 128 שכנים.
4. מכל query descriptor יישמר לכל final impression רק השכן הקרוב ביותר שהוחזר. משקל ראשוני
   הוא `exp(-0.5*(d/d128)^2)`, כאשר `d128` הוא מרחק השכן ה־128 (או epsilon אם אפס).
5. עבור כל impression בעל לפחות ארבע correspondences יותאם `estimateAffinePartial2D` עם
   RANSAC, סף 4.0 פיקסלי final, ‏2,000 iterations ו־confidence 0.995. אין homography ואין
   transform מקומי.
6. score ל־impression הוא
   `inliers * inlier_ratio * sqrt(coverage_annotated * coverage_final) * exp(-median_residual/4)`.
   coverage הוא שטח convex hull של inliers חלקי שטח התמונה, וחסר correspondences נותן 0.

זהו ערוץ Level-2 מקומי. הוא אינו קורא annotations ואינו משתמש ב־pore detector.

## 7. ערוץ O — ridge-orientation patches

1. שדה הכיוון יחושב מ־Scharr gradients לאחר Gaussian blur ‏`sigma=1.2`. בכל בלוק 8×8
   יחושבו `cos(2θ)`, ‏`sin(2θ)` ו־coherence מן structure tensor.
2. בלוקים עם coherence מתחת 0.15 יושמטו. לכל בלוק שנותר ייווצר descriptor של neighborhood
   בגודל 5×5 של שלושת הערוצים, עם reflection padding, normalization לאפס ממוצע ואורך יחידה.
   יישמרו לכל היותר 512 בלוקים, בדירוג coherence ואז `(y,x)`.
3. matching יבוצע ב־FLANN נפרד לכל run, עם אותם `trees`, ‏`checks`, ‏128 שכנים וכלל neighbor
   אחד לכל final impression כמו בערוץ K.
4. affine-RANSAC יתבצע על מרכזי הבלוקים, בסף 1.5 בלוקים במערכת final, עם 2,000 iterations
   ו־confidence 0.995.
5. score יהיה אותה נוסחה, כאשר penalty ה־residual הוא `exp(-median_residual/1.5)`.

זהו ערוץ ridge-flow צפוף שאינו משתמש ב־SIFT descriptors, minutiae או נקבוביות.

## 8. aggregation, candidates והגדרת correspondence חזק

לכל ערוץ בנפרד:

- score של identity הוא ממוצע של שלושת scores הגבוהים מבין עשרת ה־impressions, כולל אפסים;
- top-1 ו־top-2 נקבעים לפי score יורד ואז numeric identity עולה רק כ־tie-break דטרמיניסטי;
- `robust_separation = (top1-top2) / max(1.4826*MAD(all 148 scores), 1e-9)`;
- support של K דורש לפחות שני impressions עם 6 inliers, inlier ratio ‏0.15 ו־coverage בשני
  הצדדים 0.01;
- support של O דורש לפחות שלושה impressions עם 10 inliers, inlier ratio ‏0.15 ו־coverage
  בשני הצדדים 0.03;
- margin עובר כאשר `robust_separation >= 2.5` ובנוסף `top1 >= 1.25*top2`; כאשר top2 אפס,
  תנאי היחס עובר רק אם top1 חיובי;
- leave-one-out stability דורשת שה־top-1 לא ישתנה כאשר כל אחד מעשרת scores של הקבוצה המנצחת
  מושמט בתורו וממוצע top-3 מחושב מחדש.

יחידה תסווג `STRONG` רק אם top-1 של K ושל O זהה, support ו־margin עוברים בשניהם,
leave-one-out עובר בשניהם, ואין collision בין יחידות `STRONG` באותו run. `AMBIGUOUS` הוא כל
מקרה אחר; אין כפיית assignment.

לבדיקת מבנה one-to-one יחושב גם Hungarian maximum-weight assignment בכל run על הממוצע של
z-scores robust של K ו־O. assignment הוא diagnostic בלבד. יחידה חזקה דורשת שה־assignment
הגלובלי יסכים עם top-1 המקומי; assignment אינו יכול להפוך מקרה עמום לחזק.

בדיקת robustness נוספת תשמיט, עבור כל אחד מעשרת positions הקנוניים, את אותו position מכל
148 הקבוצות ותחשב מחדש winners. שיעורי שינוי, collisions ו־capture-group dependence ידווחו.

## 9. שער 1 והקפאת crosswalk

שער הזהות עובר רק אם כל התנאים שנמסרו לתכנון מתקיימים:

- לפחות 703 מתוך 740 (95%) מסווגות `STRONG`;
- אין permutation, collision או pattern שיטתי בלתי מוסבר;
- שני הערוצים מסכימים ברוב המכריע, שידווח מספרית;
- בביקורת הקפואה לפחות 45/50 הם `CLEAR` או `AMBIGUOUS`, ולפחות 38/50 הם `CLEAR`.

אם השער נכשל, geometry לא ירוץ וההכרעה תהיה `NO_RELIABLE_CROSSWALK`. אם הוא עובר,
`crosswalk.csv` יוקפא עם SHA-256 לפני כל קריאת TSV או בדיקת pore transfer.

## 10. registration לשער הגאומטריה

לכל אחד מ־60 הזוגות יבוצע matching ישיר של RootSIFT עם mutual nearest-neighbor ו־Lowe ratio
0.82. יותאמו, לפי סדר מורכבות קבוע, similarity (`estimateAffinePartial2D`) ואז affine מלא
(`estimateAffine2D`), שניהם ב־RANSAC סף 3.0 פיקסלי final. affine מלא ייבחר רק אם הוא מוסיף
לפחות 20% inliers ומפחית median residual לפחות 15% לעומת similarity; אחרת נשמר similarity.
לא יבוצעו ECC, homography או non-rigid fitting בניסוי זה.

ה־registration יסווג אוטומטית על מבנה רכסים בלבד:

- `VALID`: לפחות 12 inliers, inlier ratio לפחות 0.25, median residual עד 2.5 final pixels,
  coverage לפחות 0.04 ב־annotated ולפחות 0.08 ב־final, singular values של החלק הלינארי בין
  0.25 ל־1.25, anisotropy עד 1.6, ולפחות 0.30 cosine agreement של שדה הכיוון באזור החופף;
- `AMBIGUOUS`: transform קיים אך לפחות תנאי `VALID` אחד נכשל;
- `INVALID`: אין transform, פחות מארבעה inliers, determinant לא חיובי, או פחות 10% overlap.

לוחות ridge-only של כל 60 הזוגות יישמרו מקומית לביקורת, אך הסיווג לא ישונה לפי pores.
רק `VALID` ממשיך לדגימת נקבוביות.

## 11. tolerance ודגימת נקבוביות

אחרי הקפאת סיווגי registration, ולפני יצירת crop כלשהו סביב יעד, קואורדינטות TSV יעברו
deduplication לפי `(canonical_annotated_id,x,y)`. עד 20 נקודות לזוג ייבחרו לפי
`SHA256(l3sf-exp003-pores-v1|pair_id|x|y)`. אין החלפה לפי נראות.

ל־affine בעל חלק לינארי `A`, scale אפקטיבי הוא `sqrt(abs(det(A)))`. כלל 4/512 מניסוי 002
מומר לרדיוס final באמצעות `round_half_up(4*scale)`, עם מינימום פיקסל אחד. אלה שלושת
ה־tolerances הקפואים:

- sensitivity נמוכה: `max(1, round_half_up(3*scale))`;
- ראשית: `max(1, round_half_up(4*scale))`;
- sensitivity גבוהה: `max(1, round_half_up(5*scale))`.

נקודה מחוץ ל־[0,319]×[0,239] היא `OUT_OF_FIELD` ואינה כשל preservation. לכל נקודה אחרת
יופק crop מקומי עם marker שאינו מכסה את פיקסלי המרכז. הסוקר, ללא הזזה של prediction, יסווג
`CLEAR`, ‏`AMBIGUOUS` או `NOT_MATCH` לפי התאמת pore structure בתוך הרדיוס הראשי. בדיקות
sensitivity לא ישנו את label הראשי.

## 12. שער 2

שער geometry עובר רק אם:

- לפחות 48/60 registrations הם `VALID`;
- מבין הנקודות בתוך השדה, לפחות 85% הן `CLEAR` או `AMBIGUOUS` ולפחות 70% `CLEAR`;
- אין כשל שיטתי לפי run, pattern או capture group;
- כל fitting ו־validity נקבעו לפני קריאת annotations וללא pore-ground-truth tuning.

## 13. תוצרים, שחזור ובדיקות

תוצרים קומפקטיים יישמרו תחת `artifacts/experiment-003/`; פיקסלים תחת
`artifacts/experiment-003/evidence-pixels/`; מטריצות גדולות תחת
`artifacts/experiment-003/local-large/`. שתי התיקיות האחרונות מוחרגות מ־Git.

לכל תוצר גדול יישמר manifest עם filename, rows, bytes, SHA-256 ופקודת regeneration. קוד
reusable ייכנס ל־`src/fingerprint_new_method/`, orchestration ל־`scripts/`, וייכתבו בדיקות
dataset-independent ל־parsing, transforms, assignment, sampling, deduplication,
serialization ו־tolerance. בסיום יורצו `pip check`, ‏`pytest -q`, ‏`ruff check .` ו־`compileall`.

כל חריגה מן הפרוטוקול, לרבות כשל טכני, תתועד כתוצאה ולא תתוקן בדיעבד לאחר חשיפת המיפוי.
