import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction
import re
from textblob import TextBlob
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

VADER = SentimentIntensityAnalyzer()

NEGATIVE_OVERRIDES = [
    r"\bnot\s+(feeling\s+)?(so\s+)?(great|good|okay|ok)\b",
    r"\bnot\s+so\s+good\b",
    r"\bnot\s+well\b",
    r"\b(feel|feeling)\s+(bad|sad|awful|terrible|down)\b",
    r"\b(stressed|anxious|worried|upset|depressed)\b",
]

POSITIVE_OVERRIDES = [
    r"\bfeeling\s+(better|good|great|fine|okay|ok)\b",
    r"\bthat\s+(helped|helps)\b",
    r"\b(thanks|thank\s+you)\b",
]

class ActionAnalyzeSentiment(Action):

    def name(self) -> Text:
        return "action_analyze_sentiment"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        text = (tracker.latest_message.get("text") or "").lower().strip()

        # 1 override list (guaranteed polarity)
        for pat in NEGATIVE_OVERRIDES:
            if re.search(pat, text):
                return [SlotSet("sentiment_score", -1.0)]

        for pat in POSITIVE_OVERRIDES:
            if re.search(pat, text):
                return [SlotSet("sentiment_score", 1.0)]

        # 2 fallback to VADER
        score = VADER.polarity_scores(text)["compound"]   # –1 … +1
        return [SlotSet("sentiment_score", score)]


# --- Sentiment override keywords ---------------------------------
NEGATIVE_KEYWORDS = {
    "depressed", "depression", "sad", "hopeless", "worthless", "anxious",
    "anxiety", "stressed", "stress", "panic", "overwhelmed", "tired",
    "lonely", "miserable", "devastated", "tearful"
}
POSITIVE_KEYWORDS = {
    "happy", "joy", "great", "amazing", "fantastic", "good", "awesome",
    "excited", "grateful", "optimistic", "wonderful", "better", "improved"
}

# Varied coping strategies
COPING_STRATEGIES = {
    "breathing": {
        "name": "breathing exercise",
        "description": "Can I guide you through a quick breathing exercise to help you feel calmer?",
        "used": False
    },
    "grounding": {
        "name": "grounding technique",
        "description": "Would you like to try a grounding technique? It helps bring your focus to the present moment.",
        "used": False
    },
    "progressive_relaxation": {
        "name": "muscle relaxation",
        "description": "How about we try some progressive muscle relaxation to release tension?",
        "used": False
    },
    "journaling": {
        "name": "thought reflection",
        "description": "Sometimes writing down thoughts can help. Would you like to explore what's really bothering you?",
        "used": False
    },
    "cognitive_restructuring": {
        "name": "thought challenging",
        "description": "Let's examine those thoughts together. What specific worries are going through your mind?",
        "used": False
    }
}

def get_next_coping_strategy(tracker):
    """Get next unused coping strategy or reset if all used"""
    used_strategies = tracker.get_slot('used_strategies') or []
    
    available_strategies = [key for key, value in COPING_STRATEGIES.items() 
                          if key not in used_strategies]
    
    if not available_strategies:
        # Reset if all strategies used
        available_strategies = list(COPING_STRATEGIES.keys())
        used_strategies = []
    
    strategy_key = random.choice(available_strategies)
    strategy = COPING_STRATEGIES[strategy_key]
    
    return strategy_key, strategy, used_strategies + [strategy_key]


class ActionRespondToMood(Action):

    def name(self) -> Text:
        return "action_respond_to_mood"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        sentiment_score = tracker.get_slot('sentiment_score') or 0.0
        user_message = tracker.latest_message.get('text', '').lower()
        last_response_type = tracker.get_slot('last_response_type')
        conversation_phase = tracker.get_slot('conversation_phase') or "initial"
        crisis_level = tracker.get_slot('crisis_level') or "none"
        
        logger.debug(f"DEBUG: Sentiment score: {sentiment_score} for message: '{user_message}'")
        
        # Check for crisis keywords first
        crisis_keywords = ['hopeless', 'suicide', 'kill myself', 'hurt myself', 'end my life', 'worthless', 'want to die', 'better off dead', 'dying']
        positive_contexts = ["feeling better", "feel better", "that helped", "thanks", "thank you", "good", "happy", "ok now", "fine now", "i'm better", "feeling fine", "calmer", "relaxed"]
        
        has_crisis_keywords = any(keyword in user_message for keyword in crisis_keywords)
        has_positive_context = any(phrase in user_message for phrase in positive_contexts)
        
        # ENTER crisis if new crisis keywords appear
        if has_crisis_keywords and crisis_level == "none":
            return [
                SlotSet("conversation_phase", "crisis"),
                SlotSet("crisis_level", "high"),
                FollowupAction("action_detect_crisis")
            ]
        
        # EXIT crisis when user clearly feels better
        if crisis_level == "high" and (sentiment_score > 0.3 or has_positive_context):
            dispatcher.utter_message(text="I'm really glad you're feeling better. Remember I'm here whenever you need support.")
            return [
                SlotSet("crisis_level", "none"),
                SlotSet("conversation_phase", "closing"),
                SlotSet("last_response_type", "positive_closure")
            ]
        
        # SPECIAL HANDLING: Post-technique response
        if last_response_type in ["breathing_completed", "technique_completed"]:
            if sentiment_score > 0.1 or any(phrase in user_message for phrase in positive_contexts):
                responses = [
                    "I'm so glad that helped you feel better!",
                    "That's wonderful to hear! These techniques can be really effective.",
                    "I'm happy that made a difference for you."
                ]
            else:
                responses = [
                    "I understand it might not have helped as much as we hoped. That's okay.",
                    "Thank you for trying that with me. Different techniques work for different people."
                ]
                
            follow_ups = [
                "Is there anything else you'd like to talk about?",
                "Remember you can use these techniques whenever you need them.",
                "How are you feeling about what originally brought you here?"
            ]
            
            dispatcher.utter_message(text=random.choice(responses))
            dispatcher.utter_message(text=random.choice(follow_ups))
            
            return [
                SlotSet("last_response_type", "positive_closure"),
                SlotSet("conversation_phase", "closing"),
                SlotSet("offered_strategy", None)
            ]
        
        # Handle negative sentiment - PROPER DETECTION
        if sentiment_score < -0.2:
            empathy_responses = [
                "I hear that you're going through a tough time.",
                "That sounds really difficult to deal with.",
                "I can sense you're struggling right now.",
                "I'm sorry to hear you're not feeling well.",
                "It sounds like things are challenging for you right now."
            ]
            
            selected_response = random.choice(empathy_responses)
            dispatcher.utter_message(text=selected_response)
            
            # Get varied coping strategy
            strategy_key, strategy, updated_used = get_next_coping_strategy(tracker)
            dispatcher.utter_message(text=strategy['description'])
            
            return [
                SlotSet("previous_mood", "negative"),
                SlotSet("last_response_type", "negative_support"),
                SlotSet("offered_strategy", strategy_key),
                SlotSet("used_strategies", updated_used),
                SlotSet("conversation_phase", "support"),
                SlotSet("crisis_level", "none")
            ]
        
        # Handle positive sentiment 
        elif sentiment_score > 0.2:
            if conversation_phase in ["closing", "positive_closure"] or last_response_type == "positive_closure":
                closure_responses = [
                    "I'm really glad you're feeling better.",
                    "It's wonderful to see this positive change.",
                    "You've made great progress in our conversation."
                ]
                
                final_offers = [
                    "Feel free to reach out anytime you need someone to talk to.",
                    "Take care, and remember these techniques are always available to you.",
                    "You have the tools now - trust yourself to use them."
                ]
                
                dispatcher.utter_message(text=random.choice(closure_responses))
                dispatcher.utter_message(text=random.choice(final_offers))
                
                return [
                    SlotSet("conversation_phase", "completed"),
                    SlotSet("last_response_type", "natural_closure")
                ]
            else:
                responses = [
                    "That's wonderful to hear.",
                    "I'm glad you're experiencing some positive emotions.",
                    "It's great that you're feeling better."
                ]
                
                follow_ups = [
                    "What's contributing to these good feelings?",
                    "That's a positive step forward.",
                    "How does this compare to how you were feeling earlier?"
                ]
                
                dispatcher.utter_message(text=random.choice(responses))
                dispatcher.utter_message(text=random.choice(follow_ups))
                
                return [
                    SlotSet("last_response_type", "positive_acknowledgment"),
                    SlotSet("conversation_phase", "positive_exploration")
                ]
            
        else:  # Neutral sentiment
            neutral_responses = [
                "I'm here to listen. What's on your mind?",
                "How are you feeling right now?",
                "What would be most helpful to talk about?",
                "Can you tell me more about what's going on?"
            ]
            
            dispatcher.utter_message(text=random.choice(neutral_responses))
            return [
                SlotSet("last_response_type", "neutral_check"),
                SlotSet("conversation_phase", "exploration")
            ]


class ActionHandleExamStress(Action):

    def name(self) -> Text:
        return "action_handle_exam_stress"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        exam_responses = [
            "Exam stress is really common, especially during this time of year.",
            "I understand how overwhelming exam pressure can feel.",
            "Many students struggle with exam anxiety - you're not alone in this."
        ]
        
        dispatcher.utter_message(text=random.choice(exam_responses))
        
        # Get varied strategy for exam stress
        strategy_key, strategy, updated_used = get_next_coping_strategy(tracker)
        
        # Customize strategy for exam context
        if strategy_key == "cognitive_restructuring":
            offer = "Let's examine those exam worries. What specific thoughts are making you most anxious?"
        elif strategy_key == "journaling":
            offer = "Sometimes it helps to break down what's really worrying you about the exams. Want to explore that together?"
        else:
            offer = strategy['description']
        
        dispatcher.utter_message(text=offer)
        
        return [
            SlotSet("conversation_context", "exam_stress"),
            SlotSet("offered_strategy", strategy_key),
            SlotSet("used_strategies", updated_used),
            SlotSet("conversation_phase", "exam_support"),
            SlotSet("crisis_level", "none")  # Ensure not in crisis
        ]

# ==================================== MISSING ROUTER ACTION ====================================
class ActionProvideOfferedTechnique(Action):
    """Router action that delivers the correct technique when user says 'yes'"""
    
    def name(self) -> Text:
        return "action_provide_offered_technique"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        offered_strategy = (tracker.get_slot("offered_strategy") or "").lower()
        
        # Map strategy to action
        strategy_mapping = {
            "breathing": "action_provide_breathing_exercise",
            "grounding": "action_provide_grounding_exercise", 
            "progressive_relaxation": "action_provide_progressive_relaxation",
            "cognitive_restructuring": "action_provide_thought_exploration",
            "journaling": "action_provide_thought_exploration",
        }
        
        followup_action = strategy_mapping.get(offered_strategy)
        
        if followup_action:
            # Clear the offered strategy to prevent loops
            return [
                SlotSet("offered_strategy", None),
                FollowupAction(followup_action)
            ]
        else:
            # If no strategy offered, fall back to sentiment analysis
            return [
                FollowupAction("action_analyze_sentiment"),
                FollowupAction("action_respond_to_mood")
            ]

# ========================================================================================

class ActionProvideBreathingExercise(Action):

    def name(self) -> Text:
        return "action_provide_breathing_exercise"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        breathing_response = """Perfect! Let's do this breathing exercise together.

Find a comfortable position and follow along:

1. Breathe in slowly through your nose for 4 seconds
2. Hold your breath gently for 4 seconds  
3. Breathe out slowly through your mouth for 6 seconds
4. Let's repeat this 3 more times

Take your time... 

How are you feeling after that exercise?"""

        dispatcher.utter_message(text=breathing_response)
        
        return [
            SlotSet("last_response_type", "breathing_completed"),
            SlotSet("technique_used", "breathing")
        ]

class ActionProvideGroundingExercise(Action):

    def name(self) -> Text:
        return "action_provide_grounding_exercise"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        grounding_response = """Let's try the 5-4-3-2-1 grounding technique to bring you into the present moment:

Look around and name:
- 5 things you can SEE
- 4 things you can TOUCH
- 3 things you can HEAR
- 2 things you can SMELL
- 1 thing you can TASTE

Take your time with each step. This helps anchor you in the present moment.

How are you feeling after trying this?"""

        dispatcher.utter_message(text=grounding_response)
        
        return [
            SlotSet("last_response_type", "technique_completed"),
            SlotSet("technique_used", "grounding")
        ]

class ActionProvideProgressiveRelaxation(Action):

    def name(self) -> Text:
        return "action_provide_progressive_relaxation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        relaxation_response = """Let's try progressive muscle relaxation to release tension:

1. Start by tensing your shoulders up to your ears for 5 seconds, then release
2. Clench your fists tight for 5 seconds, then let them go
3. Scrunch your face muscles for 5 seconds, then relax
4. Tense your leg muscles for 5 seconds, then release

Notice the contrast between tension and relaxation. Your body can learn to let go of stress this way.

How does your body feel now?"""

        dispatcher.utter_message(text=relaxation_response)
        
        return [
            SlotSet("last_response_type", "technique_completed"),
            SlotSet("technique_used", "progressive_relaxation")
        ]

class ActionProvideThoughtExploration(Action):

    def name(self) -> Text:
        return "action_provide_thought_exploration"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        exploration_response = """Let's explore what's really going on in your mind. Sometimes our thoughts can make situations feel worse than they are.

Think about what's bothering you most right now, and ask yourself:
- Is this thought helping me or making things harder?
- What would I tell a friend in this exact situation?
- What's the most realistic outcome here?

Take a moment to reflect on these questions.

What comes up for you when you think about this differently?"""

        dispatcher.utter_message(text=exploration_response)
        
        return [
            SlotSet("last_response_type", "technique_completed"),
            SlotSet("technique_used", "cognitive_restructuring")
        ]

class ActionDefaultFallback(Action):

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get('text', '').lower()
        last_response_type = tracker.get_slot('last_response_type')
        offered_strategy = tracker.get_slot('offered_strategy')
        crisis_level = tracker.get_slot('crisis_level') or "none"
        conversation_phase = tracker.get_slot('conversation_phase') or "initial"
        
        # Handle technique acceptance - ROUTE TO ROUTER ACTION
        if user_message in ["yes", "sure", "ok", "okay", "yeah", "yep"] and offered_strategy:
            return [FollowupAction("action_provide_offered_technique")]
        
        # Handle technique decline
        if user_message in ["no", "not really", "no thanks", "nope"] and offered_strategy:
            alternative_responses = [
                "That's perfectly fine. Different techniques work for different people.",
                "No problem at all. Everyone has their own preferences for managing stress."
            ]
            
            follow_up = "What would feel most helpful to you right now?"
            
            dispatcher.utter_message(text=random.choice(alternative_responses))
            dispatcher.utter_message(text=follow_up)
            
            return [
                SlotSet("last_response_type", "technique_declined"),
                SlotSet("offered_strategy", None)
            ]
        
        # Handle goodbye/closure
        if any(word in user_message for word in ["bye", "goodbye", "thanks", "thank you", "ok thanks"]):
            closure_responses = [
                "You're very welcome. Take care of yourself.",
                "I'm glad I could help. Remember, these techniques are always available to you.",
                "Take care, and don't hesitate to reach out if you need support again."
            ]
            
            dispatcher.utter_message(text=random.choice(closure_responses))
            return [SlotSet("conversation_phase", "completed")]
        
        # Post-technique feedback
        if last_response_type in ["breathing_completed", "technique_completed"]:
            return [FollowupAction("action_analyze_sentiment"), FollowupAction("action_respond_to_mood")]
        
        # CRISIS HANDLING: If in crisis and looks like a name, handle as trusted person
        if crisis_level == "high" and conversation_phase in ["crisis_intervention", "safety_planning"]:
            # Check if message looks like a name (short, no question words, not common phrases)
            if (len(user_message.strip().split()) <= 3 and 
                not any(word in user_message for word in ['what', 'how', 'why', 'when', 'where', 'no', 'yes', 'help', 'support']) and
                len(user_message.strip()) > 1 and
                user_message.strip() not in ['ok', 'okay', 'sure', 'maybe', 'perhaps']):
                
                # This looks like a trusted person name - handle it
                name = user_message.strip().title()
                if "my friend" in user_message.lower():
                    name = user_message.lower().replace("my friend", "").strip().title()
                elif "my" in user_message.lower():
                    name = user_message.lower().replace("my", "").strip().title()
                
                response = f"""That's great that you have {name} in your life. Having someone you trust is really important during difficult times.

Here's what you could say to {name}:

SIMPLE APPROACH:
"Hi {name}, I'm going through a tough time and could really use someone to talk to. Are you free to chat?"

DIRECT APPROACH:
"Hey {name}, I'm struggling with some difficult feelings right now. Would you mind if I shared what's been going on?"

Remember:
- It's okay to start small and share as much as you're comfortable with
- Real friends want to help when you're struggling
- You don't have to face this alone"""

                dispatcher.utter_message(text=response)
                return [
                    SlotSet("trusted_person_name", name),
                    SlotSet("last_response_type", "trusted_person_guidance"),
                    SlotSet("crisis_level", crisis_level),  # Keep crisis level
                    SlotSet("conversation_phase", "implementation_planning")
                ]
        
        # General fallback
        responses = [
            "I want to understand better. Can you tell me more about how you're feeling?",
            "I'm here to help. What's on your mind right now?",
            "Help me understand what you're experiencing."
        ]
        
        dispatcher.utter_message(text=random.choice(responses))
        return [SlotSet("last_response_type", "general_fallback")]


class ActionDetectCrisis(Action):
    
    def name(self) -> Text:
        return "action_detect_crisis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        crisis_response = """I'm really concerned about what you've shared. Your life has value, even when it doesn't feel that way.

If you need immediate help, please call:
- Emergency: 112
- Suicide Prevention: 9152987821
- Aasra: 022-27546669

These feelings are overwhelming, but they can change. You don't have to face this alone.

Do you have someone you trust that you could talk to right now?"""

        dispatcher.utter_message(text=crisis_response)
        return [
            SlotSet("crisis_level", "high"), 
            SlotSet("conversation_phase", "crisis_intervention"),
            SlotSet("offered_strategy", None)  
        ]

class ActionProvideOngoingSupport(Action):

    def name(self) -> Text:
        return "action_provide_ongoing_support"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        response = """I'm glad you're still talking with me. That takes courage.

Let's think of someone you trust - a friend, family member, or teacher you feel safe talking to.

What's their name? I can help you think about what to say to them."""

        dispatcher.utter_message(text=response)
        return [
            SlotSet("conversation_phase", "safety_planning"),
            SlotSet("crisis_level", "high")  
        ]

class ActionHandleFamilyPressure(Action):

    def name(self) -> Text:
        return "action_handle_family_pressure"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        response = "Family pressure can be really hard, especially when it comes from people you care about."
        dispatcher.utter_message(text=response)
        
        follow_up = "How long have you been feeling this pressure from your family?"
        dispatcher.utter_message(text=follow_up)
        
        return [SlotSet("last_response_type", "family_pressure")]

class ActionStartAssessmentChoice(Action):

    def name(self) -> Text:
        return "action_start_assessment_choice"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get('text', '').lower()
        
        if any(word in user_message for word in ['depression', 'depressed', 'phq']):
            return [FollowupAction("action_start_phq9")]
        elif any(word in user_message for word in ['anxiety', 'anxious', 'gad', 'worried']):
            return [FollowupAction("action_start_gad7")]
        
        message = """I can guide you through brief mental health screenings.

I have two options:
1. Depression screening (PHQ-9) - 9 questions, takes 2-3 minutes
2. Anxiety screening (GAD-7) - 7 questions, takes 2 minutes

Which would you like to try? Just say "depression" or "anxiety"."""
        
        dispatcher.utter_message(text=message)
        return [SlotSet("last_response_type", "assessment_choice")]

class ActionStartPHQ9(Action):

    def name(self) -> Text:
        return "action_start_phq9"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        message = """Starting the depression screening. I'll ask about the past 2 weeks.

Please respond with a number:
0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day

Question 1 of 9: Over the last 2 weeks, how often have you had little interest or pleasure in doing things?

Just type the number (0, 1, 2, or 3)."""
        
        dispatcher.utter_message(text=message)
        return [SlotSet("assessment_step", 1), SlotSet("assessment_type", "phq9")]

class ActionStartGAD7(Action):

    def name(self) -> Text:
        return "action_start_gad7"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        message = """Starting the anxiety screening. I'll ask about the past 2 weeks.

Please respond with a number:
0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day

Question 1 of 7: Over the last 2 weeks, how often have you felt nervous, anxious, or on edge?

Just type the number (0, 1, 2, or 3)."""
        
        dispatcher.utter_message(text=message)
        return [SlotSet("assessment_step", 1), SlotSet("assessment_type", "gad7")]

class ActionRedirectToMentalHealth(Action):

    def name(self) -> Text:
        return "action_redirect_to_mental_health"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        responses = [
            "I'm focused on mental health support. How are you feeling today?",
            "I'm here for mental health conversations. What's on your mind emotionally?"
        ]
        
        dispatcher.utter_message(text=random.choice(responses))
        return []

class ActionAskFollowup(Action):

    def name(self) -> Text:
        return "action_ask_followup"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        follow_ups = [
            "How does that sound to you?",
            "Does any of this feel helpful?",
            "What feels most relevant right now?"
        ]
        
        dispatcher.utter_message(text=random.choice(follow_ups))
        return []

class ActionProvideProfessionalDifferentiation(Action):

    def name(self) -> Text:
        return "action_provide_professional_differentiation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        differentiation_response = """Here's what makes me different from general AI like ChatGPT:

I'm specialized for mental health support:
- Built using therapeutic principles and CBT techniques
- Designed specifically for Indian students and cultural context
- Include crisis intervention with local helplines
- Provide clinical assessments like PHQ-9 and GAD-7
- Run locally to protect your privacy completely

Unlike general AI that gives information, I provide actual therapeutic support with clinical backing and cultural understanding."""

        dispatcher.utter_message(text=differentiation_response)
        return []

class ActionHandleTrustedPerson(Action):

    def name(self) -> Text:
        return "action_handle_trusted_person"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get('text', '')
        name = user_message.strip().title()
        
        response = f"""That's great that you have {name} in your life. Having someone you trust is really important during difficult times.

Here's what you could say to {name}:

SIMPLE APPROACH:
"Hi {name}, I'm going through a tough time and could really use someone to talk to. Are you free to chat?"

DIRECT APPROACH:
"Hey {name}, I'm struggling with some difficult feelings right now. Would you mind if I shared what's been going on?"

Remember:
- It's okay to start small and share as much as you're comfortable with
- Real friends want to help when you're struggling
- You don't have to face this alone

Want help thinking about when might be a good time to reach out to {name}?"""

        dispatcher.utter_message(text=response)
        
        return [
            SlotSet("trusted_person_name", name),
            SlotSet("last_response_type", "trusted_person_guidance")
        ]
    
class ActionExitCrisis(Action):
    """Clear crisis flags when user shows clear positive sentiment."""

    def name(self) -> Text:
        return "action_exit_crisis"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(
            text="I'm really glad you're feeling better. Remember I'm here whenever you need support."
        )
        # Clear crisis flags so normal flow resumes
        return [
            SlotSet("crisis_level", "none"),
            SlotSet("conversation_phase", "closing"),
            SlotSet("last_response_type", "positive_closure"),
        ]