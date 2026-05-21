import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="VitalityAI - Healthy Core Guide", page_icon="🍏", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f4fcf7 0%, #e8f5e9 100%);
        color: #2c3e50;
    }
    .main-title {
        font-family: 'Segoe UI', system-ui, sans-serif;
        color: #2e7d32;
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #66bb6a;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 25px;
        letter-spacing: 0.5px;
    }
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 20px !important;
        padding: 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        color: #34495e !important;
    }
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.08);
        border-color: #a5d6a7 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #e8f5e9 !important;
        border: 1px solid #c8e6c9 !important;
    }
    [data-testid="stChatMessage"] p {
        color: #2c3e50 !important;
    }
    [data-testid="stChatInput"] {
        border-radius: 15px !important;
        border: 2px solid #e8f5e9 !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
    }
    [data-testid="stChatInput"] textarea {
        color: #2e7d32 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .fresh-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #81c784, transparent);
        margin: 20px 0;
        opacity: 0.6;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🍏 VITALITY HEALTHY CORE AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title"> Ask about Nutrition, Fitness & Balanced Lifestyle</p>', unsafe_allow_html=True)
st.markdown('<div class="fresh-divider"></div>', unsafe_allow_html=True)


@st.cache_resource
def load_nlp_model():
    return spacy.load("fr_core_news_sm") 

nlp = load_nlp_model()

faq_data = [
    {
        "question": "What is a balanced diet and how should I build my healthy meals or food choices?",
        "answer": "A balanced diet includes a variety of foods: 50% vegetables and fruits, 25% lean proteins (poultry, fish, tofu, legumes), and 25% whole grains (brown rice, oats, quinoa). Include healthy fats like olive oil or avocado in moderation."
    },
    {
        "question": "How much water should I drink daily for optimal hydration? Tell me about drinking water intake.",
        "answer": "Generally, aim for 2 to 2.5 liters (approx. 8-10 glasses) of water daily. You may need more if you exercise intensely, live in a hot climate, or sweat heavily."
    },
    {
        "question": "What are the best foods to eat before and after a fitness workout or sport training?",
        "answer": "Before: Eat easily digestible carbs (banana, toast) for quick energy. After: Combine protein for muscle repair with carbs to restore glycogen reserves (e.g., Greek yogurt with berries or chicken with sweet potatoes)."
    },
    {
        "question": "How many hours of sleep or rest do I need and why is sleeping important?",
        "answer": "Adults need 7 to 9 hours of high-quality sleep per night. Sleep is essential for protein synthesis, cellular repair, memory consolidation, hormone regulation, and muscle recovery."
    },
    {
        "question": "How often should I do cardio and strength training gym workouts each week?",
        "answer": "Aim for at least 150 minutes of moderate cardio (or 75 minutes of intense cardio) weekly, alongside full-body strength training sessions 2 to 3 times a week, ensuring muscle groups have 48 hours to rest."
    },
    {
        "question": "What is intermittent fasting or skipped eating windows and how does it affect the body?",
        "answer": "It is an eating pattern cycling between fasting and eating windows (like 16 hours fasting, 8 hours eating). It supports weight loss, triggers autophagy (cellular cleanup), and stabilizes blood sugar."
    },
    {
        "question": "How can I reduce sugar cravings, sweet habits or hunger naturally throughout the day?",
        "answer": "Prioritize fiber and lean protein at breakfast to avoid insulin spikes, stay well-hydrated, manage stress levels, sleep well, and swap processed treats for natural options like berries or dark chocolate."
    },
    {
        "question": "What are macronutrients, micronutrients, proteins, carbs and calories?",
        "answer": "Macronutrients provide energy and are needed in large amounts: Carbohydrates, Proteins, and Fats. Micronutrients do not provide energy but are crucial for biological functions: Vitamins and Minerals."
    },
    {
        "question": "What are some simple ways to reduce daily stress, anxiety levels or mental fatigue?",
        "answer": "Incorporate mindfulness meditation, engage in light physical activities like yoga or walks, practice deep breathing exercises, limit high caffeine intake, and disconnect from digital screens before bed."
    },
    {
        "question": "Why is stretching or flexibility important and when should I stretch my body?",
        "answer": "Stretching maintains joint flexibility and prevents injuries. Use dynamic stretching (active movements) before working out to warm up, and static stretching (holding positions) post-workout to cool down."
    },
    {
        "question": "How do I safely achieve a caloric deficit for sustainable weight loss or getting slim?",
        "answer": "A safe caloric deficit means consuming 300 to 500 calories less than your Total Daily Energy Expenditure (TDEE). Focus on high-volume, low-calorie foods like veggies, track your movement, and avoid extreme starvation diets."
    },
    {
        "question": "What is progressive overload and how does it build muscle mass or body strength?",
        "answer": "Progressive overload means gradually increasing the intensity, weight, reps, or frequency of your resistance training over time. This constant challenge forces muscle fibers to adapt, repair, and grow bigger and stronger."
    },
    {
        "question": "Tell me about coffee. How much caffeine can I safely consume per day?",
        "answer": "Up to 400mg of caffeine per day (about 4 cups of brewed coffee) is safe for most healthy adults. To prevent sleep disruption, avoid caffeine at least 6 to 8 hours before your scheduled bedtime."
    },
    {
        "question": "Can I naturally speed up my metabolic rate or metabolism burn?",
        "answer": "Yes. Building lean muscle mass through weightlifting is the best way, as muscle burns more calories at rest than fat. Staying hydrated, getting enough sleep, and eating enough protein (which has a high thermic effect) also help."
    },
    {
        "question": "What are some quick, low-calorie healthy snack options or light food?",
        "answer": "Excellent snacks include a handful of almonds or walnuts, apple slices with peanut butter, carrots with hummus, plain cottage cheese, boiled eggs, or a small bowl of mixed berries."
    },
    {
        "question": "What are the best plant-based protein sources for vegetarians and vegans?",
        "answer": "Top plant-based proteins include lentils, chickpeas, black beans, tofu, tempeh, edamame, seitan, quinoa, hemp seeds, chia seeds, and spirulina. Combining different sources ensures you get all essential amino acids."
    },
    {
        "question": "Are all dietary fats bad for health or do we need good fats?",
        "answer": "Fats are essential for hormone production, brain health, and nutrient absorption. Avoid artificial trans fats, but regularly consume healthy unsaturated fats like omega-3s found in salmon, walnuts, chia seeds, and olive oil."
    },
    {
        "question": "How can I prevent muscle soreness, stiffness or DOMS after an intense workout?",
        "answer": "Reduce Delayed Onset Muscle Soreness by staying hydrated, foam rolling, taking warm baths, getting plenty of sleep, consuming enough protein, and engaging in light active recovery like walking or swimming the next day."
    },
    {
        "question": "Which vitamins, defenses and foods strengthen the human immune system?",
        "answer": "Vitamin C (citrus fruits, bell peppers), Vitamin D (sunlight, eggs), Zinc (seeds, seafood), and antioxidants are key. Eating gut-healthy probiotic foods like kefir, kimchi, and yogurt also boosts immunity since 70% of the immune system lives in the gut."
    },
    {
        "question": "How can I stay active if I work a sedentary desk job or office office sitting all day?",
        "answer": "Try using a standing desk conversion, set an alarm to stand up and stretch every 60 minutes, take walking meetings if possible, take the stairs instead of the elevator, and use half of your lunch break for a brisk 15-minute outdoor walk."
    },
    {
        "question": "What are the signs of mental burnout and how can I recover from emotional exhaustion?",
        "answer": "Signs include chronic fatigue, irritability, detachment, and decreased performance. Recovery requires setting strict work-life boundaries, practicing saying 'no', prioritizing deep rest, and seeking professional support if needed."
    },
    {
        "question": "How does drinking water and nutrition impact skin health and acne reduction?",
        "answer": "Good hydration maintains skin elasticity and flushes toxins. Diets rich in antioxidants (berries, greens) and Omega-3 fatty acids reduce inflammation, while high-glycemic sugar foods can trigger acne flares."
    },
    {
        "question": "What are superfoods and which ones should I add to my daily diet?",
        "answer": "Superfoods are nutrient-dense items exceptionally rich in vitamins and minerals. Top examples include blueberries (antioxidants), kale and spinach (fiber/iron), chia seeds (omega-3), and turmeric (anti-inflammatory)."
    },
    {
        "question": "How can I lower my bad LDL cholesterol levels through diet naturally?",
        "answer": "Increase your intake of soluble fiber (found in oats, beans, Brussels sprouts) which reduces cholesterol absorption. Limit saturated fats from fatty meats, eliminate trans fats, and consume heart-healthy olive oil."
    },
    {
        "question": "How can I reduce my sodium or salt intake without losing food flavor?",
        "answer": "You can replace table salt with rich spices and herbs like garlic powder, onion flakes, lemon juice, smoked paprika, black pepper, cumin, or fresh herbs like basil and rosemary to enhance flavor naturally."
    },
    {
        "question": "Is doing cardio workouts on an empty stomach or fasted cardio better for fat burning?",
        "answer": "Fasted cardio can slightly increase the percentage of fat used as fuel during the exercise. However, total daily caloric deficit and overall consistency matter much more for long-term body fat loss."
    },
    {
        "question": "Why is gut health crucial and what foods improve my digestive microbiome?",
        "answer": "The gut microbiome regulates digestion, mood (serotonin production), and immunity. Boost it by eating prebiotic foods (garlic, onions, bananas) to feed good bacteria, and probiotic foods (yogurt, kefir, sauerkraut)."
    },
    {
        "question": "What is the best dietary approach to lower high blood pressure naturally?",
        "answer": "The DASH diet (Dietary Approaches to Stop Hypertension) is highly recommended. It emphasizes eating fruits, vegetables, whole grains, and lean proteins while drastically reducing sodium, red meat, and added sugars."
    },
    {
        "question": "What are the healthiest cooking techniques to preserve nutrients in food?",
        "answer": "Steaming and microwaving are excellent as they use minimal water and heat, preserving water-soluble vitamins. Roasting or air-frying with minimal olive oil are also great alternatives to deep frying."
    },
    {
        "question": "What is meal prepping and how does it help maintain healthy habits?",
        "answer": "Meal prepping is cooking full meals or portions ahead of time for the week. It saves time, reduces decision fatigue, prevents impulsive unhealthy fast-food orders, and allows strict control over ingredients and portion sizes."
    }
]

faq_questions = [item["question"] for item in faq_data]

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ if token.lemma_ else token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

preprocessed_faqs = [preprocess_text(q) for q in faq_questions]

def get_bot_response(user_query):
    clean_user_query = preprocess_text(user_query)
    if not clean_user_query.strip():
        return "Wellness Core Alert: Input is empty. Please ask a health-related question."

    vectorizer = TfidfVectorizer()
    all_texts = preprocessed_faqs + [clean_user_query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[best_match_idx]
    
    if highest_score > 0.12: 
        return faq_data[best_match_idx]["answer"]
    else:
        return "I couldn't find an exact match for that specific topic in our wellness logs. Try asking about muscle soreness, plant-based proteins, safe caloric deficit, desk jobs, coffee, or immune system!"


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Vitality AI companion. Ask me any question about nutrition, exercise, sleep, weight loss, or healthy habits!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("Ask about meals, weight loss, metabolism, coffee, sore muscles..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing wellness matrix..."):
            response = get_bot_response(user_input)
            st.write(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})