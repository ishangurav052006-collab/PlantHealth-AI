"""
PlantHealth AI - Configuration and Metadata Module
Centralizes paths, hyperparameters, dataset configuration, and disease taxonomy/recommendations.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for directory in [MODELS_DIR, RESULTS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset & Image Specs
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
CHANNELS = 3
NUM_CLASSES = 38
RANDOM_SEED = 42

# Data Splitting Strategy (Stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training Hyperparameters
BATCH_SIZE = 32
DEFAULT_EPOCHS = 15
LEARNING_RATE_CUSTOM = 1e-3
LEARNING_RATE_TRANSFER = 1e-4
LEARNING_RATE_FINE_TUNE = 1e-5

# Model Artifact Locations
CUSTOM_CNN_MODEL_PATH = MODELS_DIR / "custom_cnn.keras"
MOBILENET_MODEL_PATH = MODELS_DIR / "plantdiseasemodel.keras"
VGG16_MODEL_PATH = MODELS_DIR / "vgg16_model.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

# Evaluation Artifact Locations
EVAL_REPORT_PATH = RESULTS_DIR / "evaluation_report.md"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.png"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.png"
PREDICTION_SAMPLES_PATH = RESULTS_DIR / "prediction_samples.png"
METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"

# Canonical 38 Class Taxonomy and Agricultural Advisory
CLASS_METADATA = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "status": "Diseased",
        "pathogen": "Venturia inaequalis (Fungus)",
        "description": "Apple scab is a severe fungal disease causing dark, olive-green to black velvety spots on leaves and fruit, causing premature defoliation and fruit deformities.",
        "prevention": [
            "Rake and destroy fallen leaves in autumn to eliminate overwintering spores.",
            "Prune canopy trees to maximize sunlight penetration and air circulation.",
            "Apply protective fungicides (e.g., captan, sulfur, or copper) early in spring from green tip until petal fall."
        ]
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "status": "Diseased",
        "pathogen": "Botryosphaeria obtusa (Fungus)",
        "description": "Black rot manifests as circular brown leaf spots ('frog-eye' leaf spot) and dark sunken rot lesions on developing fruit, often accompanied by branch cankers.",
        "prevention": [
            "Prune out dead wood, fire blight cankers, and mummified apples from the orchard.",
            "Sanitize pruning shears between cuts with 70% isopropyl alcohol.",
            "Apply thiophanate-methyl or captan sprays during blossom and early cover periods."
        ]
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "status": "Diseased",
        "pathogen": "Gymnosporangium juniperi-virginianae (Fungus)",
        "description": "Produces conspicuous yellow-orange spots on upper leaf surfaces that develop tube-like fungal aecia beneath, requiring eastern red cedar as an alternate host.",
        "prevention": [
            "Remove nearby eastern red cedar and juniper trees within a 1-2 mile radius if possible.",
            "Remove galls from ornamental junipers in winter before spring spore discharge.",
            "Spray myclobutanil or mancozeb when flower buds turn pink until 30 days after petal fall."
        ]
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "The apple foliage exhibits vibrant green pigmentation, firm cuticle structure, and no signs of chlorosis, necrosis, or fungal sporulation.",
        "prevention": [
            "Maintain balanced nitrogen fertilization to avoid overly vegetative succulent growth.",
            "Continue regular monitoring for early signs of mites and aphid infestations.",
            "Ensure adequate deep watering during prolonged dry spells."
        ]
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Foliage exhibits vigorous vegetative growth, glossy waxy cuticle, and robust vascular health typical of thriving ericaceous shrubs.",
        "prevention": [
            "Maintain acidic soil pH (between 4.5 and 5.2) with elemental sulfur or peat moss.",
            "Mulch with pine bark or wood shavings to conserve root moisture.",
            "Implement drip irrigation to prevent overhead wetting of foliage."
        ]
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "status": "Diseased",
        "pathogen": "Podosphaera clandestina (Fungus)",
        "description": "Characterized by circular white, powdery fungal patches on young foliage and shoots, leading to leaf curling, distortion, and stunted terminal shoot development.",
        "prevention": [
            "Avoid excessive nitrogen applications that promote susceptible lush shoot growth.",
            "Ensure proper orchard air circulation through selective canopy pruning.",
            "Apply sulfur-based or potassium bicarbonate fungicides at petal fall and shuck split."
        ]
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Leaves are uniform green, firm, unblemished, and fully expanded without fungal patches, spots, or pest damage.",
        "prevention": [
            "Inspect regularly for early spring aphid activity.",
            "Maintain orchard sanitation and prune dead wood during dormancy.",
            "Provide consistent moisture through root zones."
        ]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Gray Leaf Spot",
        "status": "Diseased",
        "pathogen": "Cercospora zeae-maydis (Fungus)",
        "description": "Produces rectangular, tan-to-gray necrotic lesions strictly delimited by leaf veins, coalescing to blighted broad sections of photosynthetic tissue.",
        "prevention": [
            "Plant resistant corn hybrids tailored to your geographic zone.",
            "Practice crop rotation away from corn for at least one to two seasons.",
            "Till under crop residues in high-tillage systems or apply strobilurin/triazole fungicides at tasseling."
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common Rust",
        "status": "Diseased",
        "pathogen": "Puccinia sorghi (Fungus)",
        "description": "Produces small, powdery, cinnamon-brown pustules scattered across both upper and lower leaf surfaces, reducing grain filling and yield.",
        "prevention": [
            "Select resistant or tolerant corn hybrids.",
            "Plant early to avoid peak midsummer rust spore showers.",
            "Apply labeled foliar fungicides if rust pustules appear on upper leaves before tasseling."
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "status": "Diseased",
        "pathogen": "Exserohilum turcicum (Fungus)",
        "description": "Causes long, elliptical, cigar-shaped grayish-green to tan lesions that can measure up to 6 inches, devastating upper foliage during silking.",
        "prevention": [
            "Use hybrid cultivars with multi-gene (Ht) or polygenic resistance.",
            "Implement minimum 2-year crop rotation with non-host crops (soybeans, legumes).",
            "Apply fungicides at VT-R1 growth stages if threshold conditions are reached."
        ]
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Vibrant emerald green leaves with strong parallel venation, intact leaf collars, and robust chlorophyll concentration.",
        "prevention": [
            "Maintain optimal soil fertility with split nitrogen application.",
            "Scout regularly for armyworms and corn borer entry.",
            "Ensure proper drainage to prevent waterlogging."
        ]
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "status": "Diseased",
        "pathogen": "Guignardia bidwellii (Fungus)",
        "description": "Produces reddish-brown leaf spots with tiny black pycnidia; infects berries causing them to shrivel into hard, black, mummified raisins on the vine.",
        "prevention": [
            "Prune out and destroy mummified berries and infected canes during winter.",
            "Maintain open canopy management with shoot positioning and leaf pulling.",
            "Apply protective fungicides (e.g., mancozeb, captan, tebuconazole) from bud break through fruit set."
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "status": "Diseased",
        "pathogen": "Phaeomoniella chlamydospora & Fomitiporia mediterranea",
        "description": "Causes characteristic 'tiger-striped' chlorotic and necrotic patterns between leaf veins and dark purple speckling on developing berries.",
        "prevention": [
            "Protect pruning wounds with wound sealant or biocontrol agents (Trichoderma).",
            "Delay pruning until late winter when spore production is lower.",
            "Remove and incinerate severely infected vines showing apoplexy symptoms."
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis Spot)",
        "status": "Diseased",
        "pathogen": "Pseudocercospora vitis (Fungus)",
        "description": "Forms irregular, dark brown to black necrotic lesions with distinct margins on leaves, causing late-season premature leaf drop and weakening vines.",
        "prevention": [
            "Ensure adequate trellis aeration and weed suppression.",
            "Clean fallen leaves after harvest to reduce fungal inocula.",
            "Apply copper-based fungicides or broad-spectrum protectants post-bloom."
        ]
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Palmate grape foliage displays clean venation, vibrant green coloration, and robust vascular integrity without spots or mildew.",
        "prevention": [
            "Maintain appropriate trellis training and balance shoot density.",
            "Implement scheduled scoutings during warm humid spells.",
            "Apply balanced micronutrients (zinc, boron) as indicated by petiole tests."
        ]
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Citrus Greening (Huanglongbing)",
        "status": "Diseased",
        "pathogen": "Candidatus Liberibacter asiaticus (Bacterium)",
        "description": "A devastating systemic disease transmitted by Asian citrus psyllids, causing asymmetrical yellow mottling across leaf veins, dieback, and bitter misshapen fruit.",
        "prevention": [
            "Control Asian citrus psyllid vectors with systemic insecticides and biocontrol wasps.",
            "Plant only certified disease-free nursery stock from enclosed nurseries.",
            "Scout regularly and aggressively eradicate confirmed infected trees to protect orchards."
        ]
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "status": "Diseased",
        "pathogen": "Xanthomonas arboricola pv. pruni (Bacterium)",
        "description": "Causes small, angular, water-soaked purplish lesions on leaves that drop out to create a 'shot-hole' appearance, plus pitted lesions on peach fruit.",
        "prevention": [
            "Select resistant cultivars suited to humid climates.",
            "Avoid excessive late-spring nitrogen fertilization.",
            "Apply preventative copper sprays at dormant bud stages and oxytetracycline during bloom."
        ]
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Lanceolate peach leaves show smooth margins, bright deep green hue, and no shot-hole spotting, curling, or gummy exudates.",
        "prevention": [
            "Maintain proper open-center pruning for maximum air movement.",
            "Apply dormant copper sprays to prevent peach leaf curl.",
            "Mulch tree base to maintain steady moisture levels."
        ]
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "status": "Diseased",
        "pathogen": "Xanthomonas euvesicatoria (Bacterium)",
        "description": "Produces water-soaked, irregular circular spots on leaves that turn dark brown with chlorotic halos, leading to extensive blossom drop and defoliation.",
        "prevention": [
            "Use certified disease-free, hot-water-treated seed and resistant pepper varieties.",
            "Avoid overhead sprinkler irrigation; use drip lines under plastic mulch.",
            "Apply copper-mancozeb tank mixes or biofungicides (Bacillus subtilis) preventively."
        ]
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Healthy bell pepper foliage with glossy dark-green leaves, intact petioles, and active flower bud development.",
        "prevention": [
            "Maintain uniform moisture to avoid blossom end rot.",
            "Scout undersides of leaves weekly for aphids and thrips.",
            "Provide calcium and magnesium supplementation as required."
        ]
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "status": "Diseased",
        "pathogen": "Alternaria solani (Fungus)",
        "description": "Produces dark brown concentric target-like rings on older lower leaves, progressively expanding and yellowing surrounding tissue.",
        "prevention": [
            "Plant certified certified seed tubers free of Alternaria.",
            "Ensure proper nitrogen and potassium nutrition to avoid early vine stress.",
            "Apply chlorothalonil, mancozeb, or azoxystrobin fungicides at canopy closure."
        ]
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "status": "Diseased",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "description": "Highly destructive water-mold causing large, dark, water-soaked oily spots with white fluffy mildew on the leaf undersides in cool, moist weather.",
        "prevention": [
            "Plant resistant cultivars and eliminate volunteer potatoes and cull piles.",
            "Avoid prolonged leaf wetness by optimizing field row spacing and drip irrigation.",
            "Spray systemic fungicides (mefenoxam, cymoxanil, mandipropamid) when blight weather forecasts trigger alerts."
        ]
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Compound potato foliage is lush, crisp green, and free from blight lesions, hollow stems, or vascular browning.",
        "prevention": [
            "Hill up soil around plants to protect developing tubers from sun and spores.",
            "Maintain routine scouting during humid cool conditions.",
            "Rotate solanaceous crops on a 3- to 4-year cycle."
        ]
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Serrated pinnate canes with rich green coloration, vigorous primocane growth, and pristine foliage without spur blight or rust.",
        "prevention": [
            "Prune spent floricanes immediately post-harvest to enhance airflow.",
            "Trellis canes upright to avoid ground contact and soil splash.",
            "Apply dormant lime-sulfur before spring bud break."
        ]
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Trifoliate leaves show uniform bright green pigmentation, active nitrogen fixation nodulation, and absence of rust or frogeye spots.",
        "prevention": [
            "Rotate crops annually with corn or small grains.",
            "Scout for soybean cyst nematodes and fungal rusts during pod development.",
            "Ensure adequate potash levels in soil."
        ]
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "status": "Diseased",
        "pathogen": "Podosphaera xanthii (Fungus)",
        "description": "Covers upper and lower leaf surfaces, petioles, and stems with talcum powder-like fungal colonies, causing premature senescence and sunscald on fruits.",
        "prevention": [
            "Plant powdery mildew-resistant squash varieties.",
            "Provide ample spacing between hills to ensure rapid drying after dew.",
            "Apply potassium bicarbonate, neem oil, or strobilurin fungicides at first sign of colonies."
        ]
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "status": "Diseased",
        "pathogen": "Diplocarpon earlianum (Fungus)",
        "description": "Causes numerous small, irregular dark purple spots that enlarge, dry up, and turn brown, giving the leaf canopy a scorched, burnt appearance.",
        "prevention": [
            "Remove and destroy infected leaves during post-harvest renovation.",
            "Maintain narrow plant rows to facilitate good air drainage.",
            "Apply protective fungicides (captan or pyraclostrobin) during bloom periods."
        ]
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Trifoliate leaves have deep emerald gloss, intact serrated borders, and vigorous crown vitality without leaf spot or crown rot.",
        "prevention": [
            "Mulch clean straw underneath plants to prevent soil contact.",
            "Use drip irrigation rather than overhead sprinklers.",
            "Renovate beds annually after harvest."
        ]
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "status": "Diseased",
        "pathogen": "Xanthomonas vesicatoria (Bacterium)",
        "description": "Causes small, dark, greasy water-soaked spots on leaves that turn necrotic with translucent borders, causing heavy defoliation in warm wet conditions.",
        "prevention": [
            "Disinfect stakes, tools, and greenhouse flats with 10% bleach.",
            "Avoid handling plants or cultivating when foliage is wet.",
            "Apply copper bactericides combined with mancozeb to overcome copper tolerance."
        ]
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "status": "Diseased",
        "pathogen": "Alternaria linariae (Fungus)",
        "description": "Characterized by brown-to-black spots with concentric target-like rings on older leaves, surrounded by yellow chlorotic halos, progressing upward.",
        "prevention": [
            "Prune lower bottom leaves (12-18 inches from ground) to stop soil-borne spore splash.",
            "Mulch around tomato stems with clean straw or plastic film.",
            "Apply biofungicides or copper/chlorothalonil fungicides every 7-10 days in humid weather."
        ]
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "status": "Diseased",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "description": "Fast-moving water-mold causing large dark greasy patches on leaves and greasy brown lesions on stems and fruit; destroys entire fields within days.",
        "prevention": [
            "Destroy all infected plant matter immediately upon detection; do not compost.",
            "Select resistant cultivars such as 'Defiant', 'Mountain Magic', or 'Iron Lady'.",
            "Spray systemic oomycete-targeted protectants before wet cold fronts arrive."
        ]
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "status": "Diseased",
        "pathogen": "Passalora fulva (Fungus)",
        "description": "Produces pale greenish-yellow spots on upper leaf surfaces and velvety olive-brown mold on the undersides, especially prevalent in high humidity greenhouses.",
        "prevention": [
            "Improve greenhouse ventilation and install circulating fans.",
            "Maintain relative humidity below 85% through heating and venting.",
            "Apply preventative sulfur or copper sprays early in the season."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "status": "Diseased",
        "pathogen": "Septoria lycopersici (Fungus)",
        "description": "Forms numerous small circular spots with dark brown margins and sunken gray-white centers containing tiny black pycnidia, defoliating bottom-up.",
        "prevention": [
            "Rotate away from solanaceous crops for a minimum of 2-3 years.",
            "Eliminate solanaceous weeds (nightshade, horsenettle) around the perimeter.",
            "Apply protective fungicides (copper, chlorothalonil) starting at first blossom set."
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Two-Spotted Spider Mite",
        "status": "Diseased",
        "pathogen": "Tetranychus urticae (Arachnid Pest)",
        "description": "Produces fine yellow stippling and bronzing across leaves, accompanied by silky webbing on leaf undersides in hot, dry, dusty environments.",
        "prevention": [
            "Release predatory mites (Phytoseiulus persimilis) in early infestation phases.",
            "Avoid broad-spectrum synthetic pyrethroids that kill natural mite predators.",
            "Spray with insecticidal soap, horticultural oil, or abamectin targeting leaf undersides."
        ]
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "status": "Diseased",
        "pathogen": "Corynespora cassiicola (Fungus)",
        "description": "Causes circular brown lesions with light brown centers and dark concentric rings; lesions coalesce into large dead leaf sections causing rapid defoliation.",
        "prevention": [
            "Maintain open canopy architecture and stake plants securely.",
            "Discard crop debris after harvest to stop fungal overwintering.",
            "Apply fungicides such as fluxapyroxad or azoxystrobin."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "status": "Diseased",
        "pathogen": "Begomovirus (Whitefly-transmitted Virus)",
        "description": "Transmitted by whiteflies (Bemisia tabaci); results in severe plant stunting, erect cupped leaves with bright yellow margins, and complete flower drop.",
        "prevention": [
            "Install 50-mesh insect netting in greenhouses and nursery hoop houses.",
            "Control whitefly vectors using yellow sticky traps and systemic neonicotinoid/spiromesifen sprays.",
            "Plant resistant tomato hybrids (TYLCV-resistant varieties)."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Mosaic Virus",
        "status": "Diseased",
        "pathogen": "Tobamovirus (Mechanically-transmitted Virus)",
        "description": "Highly contagious virus causing mottled light/dark green mosaic foliage, strap-like 'fern-leaf' leaf distortion, and internal brown ring rot in fruit.",
        "prevention": [
            "Wash hands and tools thoroughly with milk or 10% trisodium phosphate before handling.",
            "Do not smoke or use tobacco near tomato plants (tobacco can carry the virus).",
            "Plant virus-resistant tomato hybrids with 'Tm-2^2' resistance gene."
        ]
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "None (Healthy)",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Foliage exhibits vigorous deep green compound leaves, strong stems, healthy glandular trichomes, and active flowering without lesions.",
        "prevention": [
            "Water regularly at ground level using drip irrigation.",
            "Stake and prune suckers to optimize solar exposure.",
            "Apply organic mulch to maintain steady root temperature and moisture."
        ]
    }
}
