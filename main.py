"""This file is used for the webapp integrating Depression Prediction and Stressors Exploration"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the Voting Classifier model, scaler, and encoders
with open('voting_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('encoders.pkl', 'rb') as enc_file:
    encoders = pickle.load(enc_file)

st.markdown("""
    <style>
        /* Set the background color for the whole page */
        body {
            background-color: #FCE5EF;
            color: #333333;
        }
        /* Main container background and text color */
        .stApp {
            background-color: #FCE5EF;
            color: #333333;
        }
        /* Font styles */
        .title-text {
            font-size: 36px;
            font-weight: bold;
            font-family: 'serif';
            color: #8B155C;
        }
        .subheader-text {
            font-size: 24px;
            font-family: 'serif';
            color: #F1428A;
        }
        .small-text {
            font-size: 16px;
            color: #555555;
        }
        /* Button styles */
        .stButton button {
            background-color: #F1428A;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 16px;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #C81B6A;
        }
        /* Slider color */
        .stSlider > div > div > div > div {
            background-color: #8B155C;
        }
        /* Expander Header */
        .st-expanderHeader {
            font-size: 20px;
            font-weight: bold;
            color: #F1428A;
        }
    </style>
""", unsafe_allow_html=True)

# App Title and Description
st.markdown("<div class='title-text'>Hi, we are Juggl</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader-text'>Juggling work, family, and self will no longer be a struggle.</div>", unsafe_allow_html=True)
st.markdown("<p class='small-text'>We predict risks of Depression to intervene sooner and help prevent burnout in working parents.</p>", unsafe_allow_html=True)

# Tabs for Navigation
tabs = st.tabs(["Depression Risk Prediction", "Stressors Exploration"])

with tabs[0]:
    st.markdown("### Depression Risk Prediction")
    
    # Input fields for user data
    st.markdown("#### User Information")
    
    years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
    industry = st.selectbox("Industry", encoders['Industry'].classes_)
    work_location = st.selectbox("Work Location", encoders['Work_Location'].classes_)
    hours_worked_per_week = st.number_input("Hours Worked Per Week", min_value=10, max_value=80, value=40)
    
    st.markdown("#### Work-Life Balance and Stress Indicators")
    work_life_balance_rating = st.slider("Work-Life Balance Rating (1-5)", min_value=1, max_value=5, value=3)
    stress_level = st.slider("Stress Level (1-5)", min_value=1, max_value=5, value=3)
    
    st.markdown("#### Health and Lifestyle")
    physical_activity = st.selectbox("Physical Activity", encoders['Physical_Activity'].classes_)
    sleep_quality = st.selectbox("Sleep Quality", encoders['Sleep_Quality'].classes_)
    
    # Encode categorical variables using loaded encoders
    try:
        industry_encoded = encoders['Industry'].transform([industry])[0]
        work_location_encoded = encoders['Work_Location'].transform([work_location])[0]
        physical_activity_encoded = encoders['Physical_Activity'].transform([physical_activity])[0]
        sleep_quality_encoded = encoders['Sleep_Quality'].transform([sleep_quality])[0]
    except Exception as e:
        st.error(f"Encoding Error: {e}")
        st.stop()
    
    feature_names = [
        'Years_of_Experience', 'Industry', 'Work_Location', 'Hours_Worked_Per_Week',
        'Work_Life_Balance_Rating', 'Physical_Activity',
        'Sleep_Quality', 'Stress_Level'
    ]
    
    # Convert inputs to a DataFrame with the correct feature names
    input_data = pd.DataFrame([[
        years_of_experience,
        industry_encoded,
        work_location_encoded,
        hours_worked_per_week,
        work_life_balance_rating,
        physical_activity_encoded,
        sleep_quality_encoded,
        stress_level
    ]], columns=feature_names)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Prediction Button
    if st.button("Predict Depression Risk"):
        try:
            prediction = model.predict(input_data_scaled)[0]
            probability = model.predict_proba(input_data_scaled)[0][1]  
            if prediction == 1:
                st.success(f"The individual is **at risk of Depression** with a probability of {probability:.2f}.")
            else:
                st.success(f"The individual is **not at risk of Depression** with a probability of {1 - probability:.2f}.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

with tabs[1]:
    st.markdown("""
        <style>
            /* Style for the page title */
            .title-text {
                font-size: 32px;
                font-weight: bold;
                font-family: 'serif';
                color: #8B155C;
            }
            /* Style for stressor headers */
            .stExpanderHeader {
                font-size: 20px;
                font-weight: bold;
                color: #F1428A;
            }
            /* Style for goals and actions */
            .goal-text {
                font-size: 18px;
                font-weight: semi-bold;
                color: #333333;
            }
            .action-text {
                font-size: 16px;
                color: #555555;
                margin-left: 20px;
            }
            /* Style for Favorite Goals section */
            .favorite-goals {
                background-color: #FFF2F7;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .favorite-goals-title {
                font-size: 24px;
                font-weight: bold;
                color: #8B155C;
            }
        </style>
    """, unsafe_allow_html=True)

    # Page Title
    st.markdown("<div class='title-text'>Explore Stressors and Mitigation Strategies</div>", unsafe_allow_html=True)
    st.write("Understanding common stressors and effective strategies to manage them can significantly improve your well-being.")
    if "favorite_goals" not in st.session_state:
        st.session_state.favorite_goals = []


    stressors_data = {
        "Stressor": [
            "Work-Life Imbalance",
            "Time Management Overload",
            "Lack of Self-Care",
            "Parenting Challenges",
            "Social Isolation",
            "Relationship Strain with Partner",
            "Financial Pressures",
            "Health Issues",
            "Childcare Challenges",
            "Decision Fatigue",
            "Overcommitment",
            "Perfectionism",
            "Technology Overuse",
            "Lack of Purpose"
        ],
        "Goals": [
            ["Establish Clear Work Boundaries", "Enhance Quality of Personal Time", "Create a Distinct Work Environment"],
            ["Prioritize Tasks Effectively", "Reduce Time on Low-Value Activities", "Delegate and Outsource Tasks"],
            ["Improve Physical Health", "Enhance Mental Well-being", "Ensure Adequate Rest"],
            ["Improve Parent-Child Communication", "Establish Consistent Discipline", "Enhance Family Routines"],
            ["Reconnect with Friends and Family", "Expand Social Circles", "Improve Social Skills"],
            ["Improve Communication", "Strengthen Emotional Connection", "Resolve Conflicts Constructively"],
            ["Increase Financial Awareness", "Reduce Unnecessary Expenses", "Enhance Income Streams"],
            ["Stay Proactive with Health", "Manage Chronic Conditions", "Promote Overall Wellness"],
            ["Find Reliable Childcare Solutions", "Create Backup Plans", "Optimize Existing Childcare Arrangements"],
            ["Simplify Daily Decisions", "Prioritize Important Decisions", "Implement Decision-Making Frameworks"],
            ["Develop Assertiveness", "Prioritize Commitments", "Manage Time Effectively"],
            ["Set Realistic Standards", "Practice Self-Compassion", "Focus on Progress Over Perfection"],
            ["Reduce Screen Time", "Use Technology Mindfully", "Enhance Digital Well-being"],
            ["Clarify Personal Values and Passions", "Engage in Meaningful Activities", "Reflect and Reassess Regularly"]
        ],
        "Actions": [
            [
                ["Set specific work hours and stick to them", "Disable work notifications after hours", "Communicate availability to colleagues and clients"],
                ["Engage in a hobby daily", "Practice mindfulness or meditation for 10 minutes", "Schedule regular family activities"],
                ["Designate a home office space", "Use a ritual to start/end workdays (e.g., short walk)", "Ensure proper ergonomics in your workspace"]
            ],
            [
                ["Use the Eisenhower Matrix to classify tasks", "Plan daily priorities (top three tasks each morning)", "Set time limits for tasks"],
                ["Identify time wasters by tracking activities for a week", "Limit social media use with app restrictions", "Implement 'deep work' sessions by blocking distractions"],
                ["List tasks to delegate at work/home", "Assign household chores to family members", "Hire services for tasks like cleaning"]
            ],
            [
                ["Exercise for 30 minutes at least three times a week", "Plan balanced meals ahead of time", "Stay hydrated by drinking at least 8 glasses of water"],
                ["Practice daily meditation", "Write in a gratitude journal each evening", "Schedule 'me time' for relaxation or hobbies"],
                ["Establish a sleep routine", "Create a restful environment", "Limit screen time before bed"]
            ],
            [
                ["Allocate one-on-one time with each child weekly", "Listen actively to your child's concerns", "Encourage open dialogue with open-ended questions"],
                ["Set clear rules and expectations together", "Use positive reinforcement", "Apply consequences consistently"],
                ["Create a shared family calendar", "Establish regular meal times to eat together", "Plan weekly family activities"]
            ],
            [
                ["Schedule regular calls or video chats", "Plan meetups with local friends", "Send personal messages to re-establish connections"],
                ["Join clubs or groups aligned with interests", "Attend community events to meet new people", "Participate in online communities"],
                ["Practice active listening in conversations", "Learn conversational techniques", "Volunteer for causes you care about"]
            ],
            [
                ["Schedule weekly couple meetings to discuss issues", "Practice active listening", "Use 'I' statements to express feelings"],
                ["Plan regular date nights without distractions", "Express appreciation daily", "Engage in shared interests"],
                ["Set ground rules for arguments", "Take time-outs during heated discussions", "Seek couples counseling if issues persist"]
            ],
            [
                ["Track all expenses", "Review bank statements monthly", "Set financial goals (e.g., save a specific amount)"],
                ["Cancel unused subscriptions", "Plan meals to avoid frequent eating out", "Compare prices before purchases"],
                ["Discuss raise/promotion opportunities", "Develop a side gig using personal skills", "Invest wisely with a financial advisor"]
            ],
            [
                ["Schedule regular check-ups", "Monitor health metrics", "Update vaccinations and preventive care"],
                ["Follow treatment plans as prescribed", "Keep a symptom diary", "Educate yourself about your condition"],
                ["Engage in regular physical activity", "Adopt a balanced diet", "Prioritize mental health through counseling"]
            ],
            [
                ["Research and interview providers", "Join parenting networks for recommendations", "Explore alternative options like nanny shares"],
                ["Identify emergency care options", "Establish a list of on-call sitters", "Negotiate flexible work arrangements"],
                ["Communicate regularly with caregivers", "Align schedules to reduce stress during transitions", "Provide resources to caregivers for better care"]
            ],
            [
                ["Simplify daily decisions by creating routines", "Prioritize important decisions in the morning", "Implement decision-making frameworks like pros and cons lists"],
                ["Make important decisions in the morning", "Limit daily decisions by planning ahead", "Delegate decisions when appropriate"],
                ["Use pros and cons lists for choices", "Set time limits for making decisions", "Develop standard operating procedures"]
            ],
            [
                ["Learn to politely decline additional responsibilities", "Role-play scenarios to practice saying no", "Set personal policies for accepting new tasks"],
                ["List all obligations and rank them by importance", "Eliminate non-essential activities", "Allocate buffer time between commitments"],
                ["Use a planner or calendar to schedule tasks", "Set deadlines to avoid procrastination", "Review commitments weekly to adjust as needed"]
            ],
            [
                ["Set realistic standards for tasks", "Limit time spent on tasks to prevent overworking", "Celebrate small achievements regularly"],
                ["Acknowledge mistakes as learning opportunities", "Avoid negative self-talk by reframing thoughts", "Engage in relaxation techniques"],
                ["Set incremental goals", "Track progress visually with charts or journals", "Reflect weekly on improvements made"]
            ],
            [
                ["Set daily screen time limits on devices", "Designate tech-free zones in the home", "Engage in alternative activities"],
                ["Disable non-essential notifications", "Schedule specific times for checking messages", "Reflect on tech usage with screen time reports"],
                ["Use apps that promote productivity", "Engage in online learning", "Implement night mode settings"]
            ],
            [
                ["Declutter one area of your home each week", "Use storage solutions like shelves and bins", "Implement a 'one in, one out' rule for new items"],
                ["Maintain a clean environment by setting a cleaning schedule", "Involve family members in housekeeping duties", "Hire professional cleaners periodically"],
                ["Incorporate plants into your home decor", "Use soothing colors and lighting", "Play relaxing music during downtime"]
            ]
        ]
    }

    # Convert the data into a DataFrame
    stressors_df = pd.DataFrame(stressors_data)

    # Function to display stressors with goals and actions
    def display_stressors(df):
        for index, row in df.iterrows():
            st.markdown(f"### {row['Stressor']}")
            for goal, actions in zip(row['Goals'], row['Actions']):
                goal_id = f"{row['Stressor']} - {goal}"
                checked = goal_id in st.session_state.favorite_goals
                is_favorite = st.checkbox(f" {goal}", value=checked, key=goal_id)
                if is_favorite and not checked:
                    st.session_state.favorite_goals.append(goal_id)
                elif not is_favorite and checked:
                    st.session_state.favorite_goals.remove(goal_id)
                with st.expander(f"**Goal:** {goal}"):
                    for action in actions:
                        st.markdown(f"- {action}")
            st.markdown("---")  

    with tabs[1]:

        st.markdown("## ⭐ Your Starred Goals")
        if st.session_state.favorite_goals:
            for goal_id in st.session_state.favorite_goals:
                try:
                    stressor, goal = goal_id.split(" - ", 1)
                    st.markdown(f"**{stressor}**: {goal}")
                except ValueError:
                    st.markdown(f"**Unknown Stressor**: {goal_id}")
        else:
            st.info("You haven't starred any goals yet. Star your favorite goals below.")

        st.markdown("---")  

        # Display Stressors with Goals and Actions
        display_stressors(stressors_df)