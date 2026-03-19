from langchain_aws.chat_models import ChatBedrockConverse
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from cinerag import config
import dotenv
import os

dotenv.load_dotenv()

def get_chat_model() :

    model = ChatBedrockConverse(
        model=config.CHAT_MODEL_ID,
        region_name="us-east-1"
    )
    return model

def get_query_enrichment_model():
    
    model = ChatGoogleGenerativeAI(
        model=config.QUERY_ENRICHMENT_MODEL_ID,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return model