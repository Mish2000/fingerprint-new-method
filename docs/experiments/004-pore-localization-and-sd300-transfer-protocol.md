# ניסוי 004 — למידת לוקליזציית נקבוביות והעברה קפואה ל־SD300

**סטטוס:** פרוטוקול קפוא לפני אימון משמעותי, 2026-08-29.

**ענף:** `experiment/004-pore-localization-transfer`.

**תחום הטענה:** הניסוי בוחן primitive של pore localization. הוא אינו בוחן matching של זהויות, fusion עם minutiae, שיפור recognition, זיהוי מקור סינתטי או liveness.

## 1. כללי עיוורון, נתיבים ותחום הנתונים

- כל הקוד והתוצרים ייכתבו רק תחת `fingerprint-new-method`.
- `fingerprint-datasets` נקרא בלבד. קוד הניסוי יפתור את מיקומו רק באמצעות `fingerprint_new_method.paths.dataset_path`; לא יוטמעו נתיבי dataset מוחלטים.
- אין שינוי ב־`fingerprint-benchmark`.
- ענף `L3-SF/Pore ground truth/Fingerprint Images` בגודל 512×512 וקובצי ה־TSV המקבילים הם מקור ה־supervision היחיד.
- `L3-SF/R1..R5` (`final_320`) לא ייקרא ולא ישמש בניסוי.
- SD300 אינו מקור ל־training, fine-tuning, checkpoint selection, threshold calibration, בחירת scale או שינוי preprocessing.
- עד הקפאת כל פלטי SD300 מותר לקרוא רק את `artifacts/experiment-001/selection/selection_manifest.json`. אין לקרוא את `ratings.json`, את `measurements.csv`, או נתון אחר החושף את סיווגי Experiment 001.
- IITI-HRF ו־PolyU-HRF אינם מצטרפים לניסוי גם אם הם זמינים.

קוד ההערכה ידרוש קובץ freeze מתאים לפני פתיחת split ה־test או לפני SD300. תוצאת test לא תשמש לשום שינוי. אם יימצא bug אמיתי לאחר פתיחת test, האירוע, זמן הפתיחה והתיקון לפי specification יתועדו במפורש; לא תיבחר חלופה לפי תוצאת test.

## 2. חלוקת L3-SF הקפואה

זרע החלוקה הטקסטואלי הוא:

```text
exp004-l3sf-grouped-stratified-split-v1
```

יחידת ה־leakage היא `pattern + local_index`, ללא טענה שביומטרית חמשת ה־runs הם אותה זהות. לכל pattern, כל 148 הקבוצות ידורגו בסדר עולה לפי:

```text
SHA256("exp004-l3sf-grouped-stratified-split-v1|{pattern}|{local_index}")
```

שוויון, שאינו צפוי, יוכרע לפי `local_index` מספרי. ההקצאה היא prefix ל־train, אחריו validation, ואחריו test:

| pattern | קבוצות | train | validation | test |
|---|---:|---:|---:|---:|
| right_loop | 74 | 44 | 15 | 15 |
| whorl | 60 | 36 | 12 | 12 |
| left_loop | 5 | 3 | 1 | 1 |
| plain_arch | 6 | 4 | 1 | 1 |
| tented_arch | 3 | 1 | 1 | 1 |
| **סה״כ** | **148** | **88** | **30** | **30** |

כל חמשת ה־runs וכל crop, heatmap או derivative של מקור יישארו באותה partition. ה־manifest יכיל source IDs יחסיים ל־dataset root, hashes, pattern, local index, run ו־partition; הוא לא יכיל נתיב dataset מוחלט.

## 3. Ground truth

- TSV מפורש ישירות כ־`x=column`, ‏`y=row`, ללא היסט וללא החלפת צירים.
- בתוך כל תמונה יישמר המופע הראשון בלבד של כל `(x,y)` זהה. לא תבוצע שום הסרה לפי appearance, עמימות או קרבה לשפה.
- manifest יתעד לכל תמונה ולכל partition את מספר הרשומות לפני ואחרי deduplication ואת מספר הכפילויות. סכום הכפילויות המצופה על סמך Experiment 002 הוא 241; אי־התאמה תעצור את האימון.
- target הוא heatmap יחיד. לכל נקודה נוצר Gaussian איזוטרופי עם `sigma=1.5 px`, מרכז בדיוק ב־GT, truncation ב־`4 sigma`, וחיבור peaks באמצעות pixel-wise maximum. בשפה ה־Gaussian נחתך בלבד.

## 4. Preprocessing ו־augmentation

Preprocessing קפוא לכל קלט detector:

1. פענוח grayscale בערוץ יחיד.
2. clipping לינארי לפי percentiles ‏1 ו־99 של התמונה; מקרה degenerate נהפך לאפסים.
3. המרה ל־uint8 והפעלת CLAHE עם `clipLimit=2.0`, ‏`tileGridSize=(8,8)`.
4. המרה ל־float32 בתחום `[0,1]`.

אין resize ל־512×512 של fingerprint שלם. L3-SF נשאר 512×512. SD300 עובר ridge-scale normalization ולאחריו tiled inference.

ב־train בלבד, לכל תמונה ובכל epoch, יידגם transform affine יחיד עם interpolation לינארי לתמונה ויישום אותה מטריצה בדיוק על הנקודות:

- rotation אחיד ב־`[-7°, +7°]`;
- translation אחיד ונפרד בכל ציר ב־`[-16,+16] px`;
- isotropic scale אחיד ב־`[0.95,1.05]`;
- border mode ‏`REFLECT_101` לתמונה;
- נקודות שמרכזן יצא מתחום `[0,511]²` מושמטות רק מאותו sample augmented;
- לאחר preprocessing: contrast אחיד ב־`[0.85,1.15]` סביב 0.5 ו־brightness אחיד ב־`[-0.10,+0.10]`, ואז clipping ל־`[0,1]`.

אין flip, elastic deformation, super-resolution, enhancement גנרטיבי או pore insertion.

## 5. המודל הראשי והאימון

המודל היחיד הוא U-Net fully-convolutional ללא pretrained weights:

- input/output: ערוץ grayscale אחד / ערוץ logits אחד באותה רזולוציה;
- ארבע רמות encoder ברוחבים `32,64,128,256`, bottleneck ברוחב `512`;
- בכל רמה שתי שכבות `3×3 Conv (bias=False) → GroupNorm(8) → ReLU`;
- downsampling באמצעות `2×2 MaxPool`;
- upsampling בילינארי (`align_corners=False`), ‏`1×1 Conv` לצמצום ערוצים, concatenated skip ושוב block כפול;
- output באמצעות `1×1 Conv`; ללא sigmoid בתוך המודל;
- padding שומר ממדים. אין dropout ואין רכיב pretrained.

ה־loss הוא binary focal loss על soft heatmap targets:

```text
-alpha*y*(1-p)^gamma*log(p) - (1-alpha)*(1-y)*p^gamma*log(1-p)
alpha = 0.95
gamma = 2.0
```

והוא ממוצע על כל הפיקסלים. הגדרות האימון:

- seeds: `40401`, ‏`40402`, ‏`40403`;
- optimizer: AdamW;
- learning rate התחלתי: `2e-4`;
- weight decay: `1e-4`;
- batch size פיזי: `2` תמונות מלאות;
- gradient accumulation: `2` (effective batch ‏4);
- maximum epochs: `80`;
- cosine decay עד `1e-6` לפי epoch;
- gradient norm clipping: `1.0`;
- mixed precision מותר ונרשם ב־manifest; DataLoader workers ‏0 ב־Windows;
- validation ללא augmentation אחרי כל epoch;
- checkpoint לכל seed הוא ה־epoch בעל focal loss ממוצע מינימלי על validation; בשוויון עד `1e-8` נבחר המוקדם;
- early stopping אחרי epoch ‏20 אם לא היה שיפור של לפחות `1e-5` במשך 12 epochs רצופים;
- initial weights, סדר התמונות וה־augmentation נגזרים רק מן ה־seed. יופעלו deterministic PyTorch/CUDA settings ככל שה־backend תומך, והסטטוס יירשם.

לא תתבצע architecture search ולא ישתנו hyperparameters בין seeds.

## 6. Post-processing ו־baseline

למודל, לאחר בחירת שלושת ה־checkpoints, ייבדק על validation בלבד grid קפוא:

- sigmoid threshold: ‏`0.05,0.06,...,0.95`;
- candidate local maxima בחלון `3×3`;
- greedy Euclidean NMS ברדיוס מתוך `{2,3,4} px`, בסדר confidence יורד ואז `y,x` עולים;
- plateau בעל ערך זהה מצטמצם דטרמיניסטית לנקודה הלקסיקוגרפית הראשונה לפני NMS.

ייבחר **threshold ורדיוס NMS משותפים לשלושת ה־seeds** לפי median ‏`F1@4px` על validation. tie-breaks, לפי הסדר: minimum seed F1 גבוה יותר, median precision גבוה יותר, threshold גבוה יותר, radius קטן יותר. לאחר כתיבת freeze manifest לא ישתנו פרמטרים אלה.

ה־baseline היחיד הוא bright local-extrema heuristic. על התמונה שעברה preprocessing יחושב:

```text
response = image - GaussianBlur(image, sigma)
```

מועמדים הם local maxima ‏3×3 מעל percentile תוך־תמונתי של response, ואחריהם אותו greedy NMS. ה־grid הקפוא הוא:

- Gaussian sigma מתוך `{1.0,1.5,2.0}`;
- percentile מתוך `{98.0,98.5,99.0,99.25,99.5,99.75}`;
- NMS radius מתוך `{2,3,4}`.

הצירוף ייבחר לפי aggregate validation ‏`F1@4px`; tie-breaks: precision, פחות FP/image, percentile גבוה יותר, sigma קטן יותר, radius קטן יותר. אין התאמה ל־test.

## 7. Matching ומדדי L3-SF

התאמת prediction↔GT היא bipartite one-to-one: תחילה maximum cardinality בתוך tolerance, ובין פתרונות בעלי אותה cardinality מינימום סכום מרחקים אוקלידיים. המימוש ישתמש ב־linear assignment עם penalty שמבטיח קדימות cardinality וייבדק מול מקרים קטנים exhaustively.

המדד הראשי הוא `F1@4px`. ידווחו גם precision, recall, mean localization error של matches, FP/image ו־recall/image. sensitivity תדווח ב־2 ו־6 px ללא שינוי הכרעה. הפירוט יהיה aggregate, לפי run, pattern, seed ותמונה.

annotation מוגדר "קרוב לשפה" אם `min(x,y,511-x,511-y) <= 8 px`. edge subset ידווח בנפרד אך יישאר בתוך המדד הראשי.

## 8. שער A והקפאה

Gate A מחושב רק לאחר freeze מפורש של שלושת ה־checkpoints, preprocessing, threshold, NMS וה־baseline.

`STRONG_PASS` דורש, עבור median של שלושת ה־seeds:

1. `F1@4px >= 0.80`;
2. precision ו־recall כל אחד `>=0.75`;
3. יתרון F1 של לפחות `0.10` absolute מעל baseline;
4. median F1 בין seeds בכל run הוא `>=0.70`;
5. `max(seed F1)-min(seed F1) <= 0.05`.

`CONDITIONAL` דורש median `F1@4px >=0.65` וכל שלושת ה־seeds מעל baseline, אך לפחות תנאי STRONG אחד אינו מתקיים. כל מצב של median F1 מתחת 0.65, או seed שאינו טוב מה־baseline, הוא `FAIL`. במצב FAIL ייקבע `LOCALIZATION_FAIL`, לא תיקרא תמונת SD300, והניסוי ייעצר ללא architecture search.

ב־STRONG_PASS או CONDITIONAL ייכתב `model_manifest.json` הכולל config hash, weights/checkpoint hashes וגדלים, seeds, versions, preprocessing, target, threshold, NMS, coordinate conventions ופקודת regeneration. קובצי weights ו־heatmaps נשארים local ואינם חייבים להיכנס ל־Git.

## 9. Ridge-scale normalization ל־SD300

אומדן ridge period אינו משתמש ב־pore detector:

1. לאחר robust percentile normalization בלבד, נלקחים tiles בגודל 256 וב־stride ‏128.
2. tile נדחה אם סטיית התקן שלו מתחת 8 gray levels או structure-tensor coherence מתחת 0.20.
3. orientation דומיננטי נגזר מה־structure tensor. ה־tile מסובב כך שהרכסים אנכיים, crop מרכזי 192×192 ממוצע לאורך הרכסים לפרופיל 1D, וה־autocorrelation המנורמל שלו נבדק ב־lags ‏5–64 px.
4. נבחר ה־local maximum הקטן ביותר שנמצא בתוך 90% מן המקסימום הגלובלי בטווח. tile אמין אם peak correlation לפחות 0.15.
5. אומדן תמונה הוא median של לפחות 5 tiles; הוא אמין אם `MAD/median <= 0.25`.

ridge period המטרה הוא median אומדני התמונות האמינות ב־L3-SF **train בלבד** וייכתב ל־freeze manifest לפני כל SD300 inference. לכל SD300 image scale factor הוא `target_period / estimated_period`; factor מחוץ `[0.20,1.50]` או אומדן לא אמין מסומן `PREPROCESSING_FAILURE`. resize משתמש ב־`INTER_AREA` להקטנה וב־`INTER_CUBIC` להגדלה. שום pore coordinate, density או repeatability אינו משתתף בבחירת scale.

## 10. Tiled inference

- tile: ‏512×512;
- overlap: ‏64 px; stride ‏448;
- תמונה קטנה מרופדת ב־`REFLECT_101` ומתועדת;
- heatmaps מתחברים במרחב התמונה המנורמל באמצעות separable cosine ramps ברוחב 32 px באזורים פנימיים וסכום משוקלל;
- threshold/local maxima/NMS מופעלים פעם אחת על ה־heatmap המחובר, ולכן overlap אינו מכפיל detections;
- נשמר transform מלא בין coordinates מקוריים לבין normalized coordinates.

שחזור coordinates, blending, padding ו־deduplication ייבדקו על heatmaps סינתטיים שאינם תלויים ב־dataset.

## 11. Pair manifest ו־registration ב־SD300

המדגם הוא בדיוק 20 הרשומות ב־selection manifest הקפוא של Experiment 001. הניתוח הראשי הוא 1000 ppi. לכל plain במיקום `i`:

- mated roll: אותו subject ואותו `i`;
- non-mated roll: אותו subject ומיקום `i+1`, וב־10 חזרה ל־1.

המיפוי ייכתב וייחתם לפני detector inference. אם ה־roll השלילי הדרוש אינו קיים, הדבר יתועד ולא תיבחר חלופה.

ה־registration אינו רואה heatmap או detection:

1. משתמש בתמונות ridge-scale-normalized שעברו Gaussian blur עם `sigma=2.0` כדי לדכא detail קטן;
2. SIFT עם `nfeatures=12000`, ‏`contrastThreshold=0.02`, ‏`edgeThreshold=10`, ‏`sigma=1.6`;
3. mutual 2-NN ratio ‏0.78;
4. homography ‏plain→roll באמצעות RANSAC: threshold ‏6 normalized px, confidence ‏0.999, maximum iterations ‏10000;
5. refinement מקומי עם Farneback flow על ridge-blurred roll הממופה למישור plain: `pyr_scale=.5`, ‏4 levels, ‏winsize 31, ‏5 iterations, ‏poly_n 7, ‏poly_sigma 1.5, Gaussian flag;
6. forward/backward flow consistency עד 2 px מגדירה pixels אמינים.

סטטוס נקבע ונשמר לפני detector scoring:

- `VALID`: לפחות 20 mutual matches, לפחות 12 RANSAC inliers, inlier ratio לפחות 0.25, median reprojection error עד 4 px, p90 עד 8 px, ו־convex-hull coverage של inlier coordinates לפחות 8% בשתי התמונות;
- `AMBIGUOUS`: לפחות 8 mutual matches ו־6 inliers אך תנאי VALID אינו מלא;
- אחרת `INVALID`.

אין תיקון אנושי. mutual overlap הוא intersection של foreground ridge masks, תחום homography, convex hulls של inliers ו־forward/backward-consistent flow. רק detections שמיפוין בתוך overlap משתתפות; detections מחוץ לו אינן FP. Δ מוגדר רק כאשר גם mated וגם non-mated registration של אותה רשומת plain הם VALID. כל זוג, כולל failures ו־zero-detection, נשאר בדוח.

matching ב־SD300 משתמש באותו one-to-one rule וב־4 px במרחב ה־ridge-normalized. אם `M` matches ו־`N1,N2` detections בתוך overlap:

```text
Repeatability = 2M / (N1 + N2)
```

`N1=N2=0` מסומן `ZERO_DETECTIONS_BOTH` ואינו הצלחה. לכל finger ו־seed: `Delta = repeatability_mated - repeatability_non_mated`. במקביל ידווחו detections/megapixel, detections/estimated-ridge-area ו־detections בתוך overlap. foreground/ridge area ייאמד ב־Otsu על התמונה המנורמלת, closing אליפטי 31×31, dilation ‏9×9 ושמירת רכיבים ששטחם לפחות 1% מן התמונה.

## 12. Gate B, bootstrap ו־2000 ppi

המדד הראשי הוא median Delta על paired VALID records. bootstrap paired percentile CI משתמש ב־10,000 resamples של fingers עם replacement וב־seed `4049001`; ה־CI הוא percentiles ‏2.5 ו־97.5. resampling זהה לכל שלושת ה־seeds.

`TRANSFER_PASS` דורש:

1. לפחות 15 מתוך 20 registrations mated הם VALID;
2. median Delta לפחות `+0.10`;
3. הגבול התחתון של paired bootstrap 95% CI גדול מ־0;
4. median Delta חיובי לפחות בשניים משלושת ה־seeds;
5. רק לאחר unblinding: לפחות 8 מתוך 11 אצבעות `LEVEL3_USABLE` בעלות median-seed Delta חיובי.

אם Gate A עבר אך registrations אינם מספיקים, האפקט אינו עובר את הספים או שונות ה־seeds גדולה, התוצאה היא `TRANSFER_INCONCLUSIVE`. אם יש לפחות 15 paired VALID אך median Delta אינו חיובי או mated אינו טוב בעקביות, התוצאה היא `TRANSFER_FAIL`.

רק לאחר חתימת פלטי 1000 ppi תרוץ sensitivity ב־2000 ppi עם אותו detector וכלל preprocessing קפוא. ידווחו density, same-impression ‏1000↔2000 repeatability וכיוון plain↔roll; שתי הרזולוציות אינן דגימות biometric עצמאיות. רק לאחר חתימת כל פלטי SD300 ייפתחו ratings של Experiment 001 לצורך התנאי החמישי ותיאור של שלוש קבוצות האיכות. שום output לא ישתנה לאחר מכן.

## 13. תוצרים, tests והכרעה

התוצרים הקומפקטיים תחת `artifacts/experiment-004/` יהיו לפחות:

```text
split_manifest.json
ground_truth_dedup_summary.json
model_protocol.json
training_runs.json
validation_summary.json
test_metrics.json
test_per_image.csv
baseline_metrics.json
model_manifest.json
sd300_pair_manifest.json
sd300_registration_summary.csv
sd300_repeatability.csv
sd300_transfer_summary.json
summary.json
```

תוצר שלא הורץ עקב שער עצירה יישמר כ־JSON/CSV קומפקטי עם status מפורש, ולא יכיל תוצאת dataset מומצאת. weights, checkpoints, heatmaps וראיות pixel-bearing יישארו בנתיבים local ignored, עם relative path, size, SHA-256, seed, config ID ופקודת regeneration ב־manifest.

לפני שימוש מדעי יתווספו tests dataset-independent עבור grouping/split, dedup, affine coordinates, Gaussian target, local maxima, NMS, optimal matching, metrics ב־2/4/6 px, tiled reconstruction, ridge period/scale, negative mapping, overlap, repeatability, deterministic bootstrap ו־manifest round-trip.

ההכרעה הסופית תהיה אחת בלבד:

- `LOCALIZATION_FAIL` אם Gate A נכשל;
- `SYNTHETIC_LOCALIZATION_ONLY` אם localization עבר אך transfer נכשל או אינו מכריע;
- `REAL_TRANSFER_SUPPORTED` רק אם Gate A ו־Gate B עברו.

לאחר ההכרעה הניסוי נעצר; לא מתחיל matcher, descriptor, fusion או synthetic detector.
