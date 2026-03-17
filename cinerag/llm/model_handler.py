from langchain_aws.chat_models import ChatBedrockConverse

def get_chat_model() :

    model = ChatBedrockConverse(
        model="amazon.nova-lite-v1:0",
        region_name="us-east-1"
    )
    return model