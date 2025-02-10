from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    InputRequiredEvent,
    HumanResponseEvent,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel, Field
from llm_imp import llm
from prompt import *
class Observation(BaseModel):
    """Observations"""
class UserGoalsFeedback(BaseModel):
    flag:bool=Field(description="Return true if user is satsfied by the system result else false")
    changes:str=Field(description="Changes the user wanted to add the in the system result.Don't change the request just correct the grmatical mistakes")
class ProgressEvent(Event):
    msg: str
class RetryGoals(Event):
    re_request:str
class ObjectiveClear(Event):
    objective:str


class CodeMate(Workflow):
    def __init__(self,max_steps:int=3, **kwargs):
        super().__init__(**kwargs)
        self.llm=llm
        self.max_steps=max_steps
    
    @step
    async def goals(
        self,ctx:Context,ev:StartEvent|RetryGoals
    )->ObjectiveClear|RetryGoals:
        
        if isinstance(ev,StartEvent):
            await ctx.set("query",ev.input)
            await ctx.set("observation_count",1)
            ctx.write_event_to_stream(ProgressEvent(msg="Understanding Your Goals"))
            generator = await self.llm.astream_complete(
                "You are given a project objective and idea from the user. "
                "Your task is to summarize the user's idea in your own words while maintaining its core intent. "
                "Ensure the summary is clear, concise, and accurately reflects the original idea.\n\n"
                "Additionally, determine whether the user is a beginner or an expert in the given field. "
                "If the user is a beginner, structure the summary in simpler terms. "
                "If the user is an expert, use more technical terminology where appropriate.\n\n"
                f"User Query/Idea:\n{ev.input}\n"
            )
            async for response in generator:
            # Allow the workflow to stream this piece of response
                ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
            await ctx.set("objectives",str(response))
            ctx.write_event_to_stream(ProgressEvent(msg="Did I understood you idea or do you want to add more. If the explanation is ok then say ok"))
            human_input=(
                "User original query -\n\n{input}\n\n"
                "System result -\n\n{response}\n\n"
                "Check if the user is ok with the system result or he wanted changes -\n"
                "This the user response -\n"
            )
            user_feedback=input("Write your feedback on the objectives, do you want to change something.")
            check=self.llm.structured_predict(
                UserGoalsFeedback,
                PromptTemplate(human_input+user_feedback),
                input=ev.input,
                response=str(response),
            )
            if check.flag:
                return ObjectiveClear(objective=str(response))
            else:
                return RetryGoals(re_request=check.changes,original=user_feedback)
        if isinstance(ev,RetryGoals):
           observation_count=await ctx.get("observation_count")
           objective=await ctx.get("objectives")
           if observation_count>3:
                
                return ObjectiveClear(objective=objective)
           else:
                original_query=await ctx.get("query")
                generator = await self.llm.astream_complete(
                    "You are given a project objective and idea from the user. "
                    "Your task is to summarize the user's idea in your own words while maintaining its core intent. "
                    "Ensure the summary is clear, concise, and accurately reflects the original idea.\n\n"
                    "Additionally, determine whether the user is a beginner or an expert in the given field. "
                    "If the user is a beginner, structure the summary in simpler terms. "
                    "If the user is an expert, use more technical terminology where appropriate.\n\n"
                    f"User original query Query/Idea:\n{original_query}\n"
                    f"Previous System response: \n{objective}\n"
                    f"Changes user asked in system response :\n{ev.re_request}"
                )
                async for response in generator:
                # Allow the workflow to stream this piece of response
                    ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
                await ctx.set("objectives",str(response))
                ctx.write_event_to_stream(ProgressEvent(msg="Did I understood you idea or do you want to add more. If the explanation is ok then say ok"))
                human_input=(
                    "User original query -\n\n{input}\n\n"
                    "System result -\n\n{response}\n\n"
                    "Check if the user is ok with the system result or he wanted changes -\n"
                    "This the user response -\n"
                )
                user_feedback=input("Write your feedback on the objectives, do you want to change something.")
                check=self.llm.structured_predict(
                    UserGoalsFeedback,
                    PromptTemplate(human_input+user_feedback),
                    input=ev.input,
                    response=str(response),
                )
                observation_count+=1
                await ctx.set("observation_count",observation_count)
                if check.flag:
                    return ObjectiveClear(objective=str(response))
                else:
                    return RetryGoals(re_request=check.changes,original=user_feedback)

           
           


        
        




