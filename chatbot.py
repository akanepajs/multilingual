from anthropic import AsyncAnthropic, Anthropic
import google.generativeai
import google.generativeai.generative_models
import openai
import google
from openai import OpenAI
from itertools import count, cycle, islice
from datetime import datetime
from copy import deepcopy
import json
from io import BytesIO
from enum import Enum
from abc import ABC, abstractmethod
from os.path import isfile
from time import perf_counter
import aiohttp
import asyncio
import re
from more_itertools import chunked, pairwise
import time
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, Iterable, TypeVar, Callable

def json_load(filename: str) -> str | int | list | dict:
    with open(filename, "r") as f:
        return json.load(f)

def json_dump(object: str | int | list | dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(object, f)

@dataclass
class APIPricingForType:
    dollars_per_million_input_tokens: float
    dollars_per_million_output_tokens: float
    
    def completion_price_dollars(self, n_input_tokens: int, n_output_tokens: int) -> float | None:
        
        return n_input_tokens  * self.dollars_per_million_input_tokens  / 1_000_000 \
             + n_output_tokens * self.dollars_per_million_output_tokens / 1_000_000
    
    def fine_tuning_training_price(self, n_training_tokens: int) -> float | None:
        return None

@dataclass
class APIPricing:
    normal: APIPricingForType | None = None
    fine_tuned: APIPricingForType | None = None
    with_batch_api: APIPricingForType | None = None
    with_batch_api_and_fine_tuned: APIPricingForType | None = None
    dollars_per_million_fine_tuning_training_tokens: float | None = None
    dollars_per_million_fine_tuning_training_tokens_with_batch_api: float | None = None

    def completion_price_dollars(self, n_input_tokens: int,
                                       n_output_tokens: int,
                                       fine_tuned: bool,
                                       batch_api: bool ) -> float | None:
        
        if not fine_tuned and not batch_api:
            if self.normal is None:
                return None
            return self.normal.completion_price_dollars(n_input_tokens, n_output_tokens)
        
        elif fine_tuned and not batch_api:
            if self.fine_tuned is None:
                return None
            return self.fine_tuned.completion_price_dollars(n_input_tokens, n_output_tokens)
        
        elif not fine_tuned and batch_api:
            if self.with_batch_api is None:
                return None
            return self.with_batch_api.completion_price_dollars(n_input_tokens, n_output_tokens)
        
        elif fine_tuned and batch_api:
            if self.with_batch_api_and_fine_tuned is None:
                return None
            return self.with_batch_api_and_fine_tuned.completion_price_dollars( n_input_tokens,
                                                                                n_output_tokens )
        
        else:
            assert False, "unreachable"

    def fine_tuning_price_dollars(self, n_training_tokens: int, batch_api: bool) -> float | None:
        if batch_api:
            dollars_per_million_training_tokens = \
                self.dollars_per_million_fine_tuning_training_tokens_with_batch_api
        else:
            dollars_per_million_training_tokens = \
                self.dollars_per_million_fine_tuning_training_tokens

        if dollars_per_million_training_tokens is None:
            return None
        return n_training_tokens * dollars_per_million_training_tokens / 1_000_000

ANTHROPIC_API_PRICING = {
    r"claude-3-5-sonnet.*": APIPricing(normal=APIPricingForType(
        dollars_per_million_input_tokens  =  3.00,
        dollars_per_million_output_tokens = 15.00
    )),
    r"claude-3-opus.*":     APIPricing(normal=APIPricingForType(
        dollars_per_million_input_tokens  = 15.00,
        dollars_per_million_output_tokens = 75.00
    )),
    r"claude-3-sonnet.*":   APIPricing(normal=APIPricingForType(
        dollars_per_million_input_tokens  =  3.00,
        dollars_per_million_output_tokens = 15.00
    )),
    r"claude-3-haiku.*":    APIPricing(normal=APIPricingForType(
        dollars_per_million_input_tokens  =  0.25,
        dollars_per_million_output_tokens =  1.25
    ))
}

OPENAI_API_PRICING = {
    r"gpt-4o.*": APIPricing(
        normal                    = APIPricingForType( dollars_per_million_input_tokens  =  5.00,
                                                       dollars_per_million_output_tokens = 15.00 ),
        with_batch_api            = APIPricingForType( dollars_per_million_input_tokens  =  2.50,
                                                       dollars_per_million_output_tokens =  7.50 )
    ),
    r"gpt-4-turbo.*": APIPricing(
        normal                    = APIPricingForType( dollars_per_million_input_tokens  = 10.00,
                                                       dollars_per_million_output_tokens = 30.00 ),
        with_batch_api            = APIPricingForType( dollars_per_million_input_tokens  =  5.00,
                                                       dollars_per_million_output_tokens = 15.00 )
    ),
    r"gpt-4": APIPricing(
        normal                    = APIPricingForType( dollars_per_million_input_tokens  = 30.00,
                                                       dollars_per_million_output_tokens = 60.00 ),
        with_batch_api            = APIPricingForType( dollars_per_million_input_tokens  = 15.00,
                                                       dollars_per_million_output_tokens = 30.00 )
    ),
    r"gpt-4-32k": APIPricing(
        normal                    = APIPricingForType( dollars_per_million_input_tokens  =  60.00,
                                                       dollars_per_million_output_tokens = 120.00 ),
        with_batch_api            = APIPricingForType( dollars_per_million_input_tokens  =  30.00,
                                                       dollars_per_million_output_tokens =  60.00 )
    ),
    r"gpt-4-.*-preview": APIPricing(
        normal                    = APIPricingForType( dollars_per_million_input_tokens  = 10.00,
                                                       dollars_per_million_output_tokens = 30.00 ),
        with_batch_api            = APIPricingForType( dollars_per_million_input_tokens  =  5.00,
                                                       dollars_per_million_output_tokens = 15.00 )
    ),
    r"gpt-3.5-turbo(?!-instruct)(?!-16k).*": APIPricing(
        normal                    = APIPricingForType( dollars_per_million_input_tokens  = 0.50,
                                                       dollars_per_million_output_tokens = 1.50 ),
        fine_tuned                = APIPricingForType( dollars_per_million_input_tokens  = 3.00,
                                                       dollars_per_million_output_tokens = 6.00 ),
        with_batch_api            = APIPricingForType( dollars_per_million_input_tokens  = 0.25,
                                                       dollars_per_million_output_tokens = 0.75 ),
        with_batch_api_and_fine_tuned \
                                  = APIPricingForType( dollars_per_million_input_tokens  = 1.50,
                                                       dollars_per_million_output_tokens = 3.00 ),
        dollars_per_million_fine_tuning_training_tokens = 8.00
    ),
    r"gpt-3.5-turbo-16k-.*": APIPricing(
        normal                    = APIPricingForType( dollars_per_million_input_tokens  =  3.00,
                                                       dollars_per_million_output_tokens =  4.00 ),
        with_batch_api            = APIPricingForType( dollars_per_million_input_tokens  =  1.50,
                                                       dollars_per_million_output_tokens =  2.00 )
    ),
    r"gpt-3.5-turbo-instruct": APIPricing(
        normal                    = APIPricingForType( dollars_per_million_input_tokens  = 1.50,
                                                       dollars_per_million_output_tokens = 2.00 ),
        with_batch_api            = APIPricingForType( dollars_per_million_input_tokens  = 0.75,
                                                       dollars_per_million_output_tokens = 1.00 )
    ),
    r"davinci-002": APIPricing(
        normal                    = APIPricingForType( dollars_per_million_input_tokens  =  2.00,
                                                       dollars_per_million_output_tokens =  2.00 ),
        fine_tuned                = APIPricingForType( dollars_per_million_input_tokens  = 12.00,
                                                       dollars_per_million_output_tokens = 12.00 ),
        with_batch_api            = APIPricingForType( dollars_per_million_input_tokens  = 1.00,
                                                       dollars_per_million_output_tokens = 1.00 ),
        with_batch_api_and_fine_tuned \
                                  = APIPricingForType( dollars_per_million_input_tokens  = 6.00,
                                                       dollars_per_million_output_tokens = 6.00 ),
        dollars_per_million_fine_tuning_training_tokens = 6.00
    ),
    r"babbage-002": APIPricing(
        normal                    = APIPricingForType( dollars_per_million_input_tokens  = 0.40,
                                                       dollars_per_million_output_tokens = 0.40 ),
        fine_tuned                = APIPricingForType( dollars_per_million_input_tokens  = 1.60,
                                                       dollars_per_million_output_tokens = 1.60 ),
        with_batch_api            = APIPricingForType( dollars_per_million_input_tokens  = 0.20,
                                                       dollars_per_million_output_tokens = 0.20 ),
        with_batch_api_and_fine_tuned \
                                  = APIPricingForType( dollars_per_million_input_tokens  = 0.80,
                                                       dollars_per_million_output_tokens = 0.80 ),
        dollars_per_million_fine_tuning_training_tokens = 0.40
    )
}

class DictMessage(TypedDict):
    role: Literal["system", "user", "content"]
    content: str

class Role(Enum):
    user = 0
    assistant = 1
    system = 2

    def opposite(self) -> "Role":
        if self == Role.system:
            raise ValueError( "Can only take the opposite of a role which is `user` or "
                              "`assistant`, not `system`." )
        elif self == Role.user:
            return Role.assistant
        elif self == Role.assistant:
            return Role.user
        else:
            assert False, "this code is unreachable"

TupleMessage = tuple[Role, str]

Message = DictMessage | TupleMessage

DictConversation  = list[DictMessage]
TupleConversation = list[TupleMessage]
Conversation      = list[Message]

class Batching(ABC):
    @abstractmethod
    def list_to_batch(self, xs: list[Any]) -> Any:
        pass

class NoBatching(Batching):
    def list_to_batch(self, xs: list[Any]) -> Any:
        assert len(xs) == 1
        return xs[0]

class ListBatching(Batching):
    def list_to_batch(self, xs: list[Any]) -> list[Any]:
        return xs

@dataclass
class DictBatching(Batching):
    keys: list[Any]

    def list_to_batch(self, xs: list[Any]) -> dict[Any, Any]:
        assert len(xs) == len(self.keys)
        return dict(zip(self.keys, xs))

class Chatbot(ABC):
    @abstractmethod
    def _completion(self, messages: Conversation,
                          max_new_tokens: int,
                          temperature: float ) -> list[str]:
        pass
    
    def __call__(self, messages_or_prompt:   Conversation
                                             | list[Conversation]
                                             | dict[Any, Conversation]
                                             | str
                                             | list[str]
                                             | dict[any, Conversation],
                         max_new_tokens:     int = 1024,
                         temperature:        float = 1.,
                         batch_size:         int = 64,
                         save_to:            str | None = None,
                         progress_bar:       bool = False ) -> str | list[str] | dict[any, str]:
        
        batching, conversations = \
            Chatbot._messages_or_prompt_to_batch_of_dict_conversations(messages_or_prompt)

        for conversation in conversations:
            Chatbot._validate_message_order(conversation)

        if save_to is not None:
            # TO DO: FACTOR THIS INTO A FUNCTION

            # this is the easiest way to assert that the file can be created before running the
            #  model
            if not isfile(save_to):
                json_dump({}, save_to)
            saved_completions = json_load(save_to)
            if self._model_name not in saved_completions.keys():
                saved_completions[self._model_name] = []

            unsaved_conversations = [
                conversation
                for conversation in conversations
                if not any( conversation == conversation_
                            for conversation_, completion in saved_completions[self._model_name] )
            ]

            new_completions = self(
                unsaved_conversations,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                batch_size     = batch_size,
                progress_bar   = progress_bar
            )

            all_completions = saved_completions[self._model_name] \
                            + list(zip(unsaved_conversations, new_completions))

            saved_completions[self._model_name] = all_completions
            if len(new_completions) != 0:
                json_dump(saved_completions, save_to)
            
            completions = [
                next(iter( completion
                           for conversation_, completion in all_completions
                           if conversation_ == conversation ))
                for conversation in conversations
            ]
            
            completions = batching.list_to_batch(completions)

            return completions

        completions = []
        # convert chunked to list for the tqdm to know the total number of steps
        batches = list(chunked(conversations, batch_size))
        for conversation_batch in tqdm(batches, desc="chatbot") if progress_bar else batches:
            completion_batch = self._completion( conversation_batch,
                                                 max_new_tokens = max_new_tokens,
                                                 temperature = temperature )
            completions += completion_batch
        
        completions = batching.list_to_batch(completions)

        return completions

    @staticmethod
    def _messages_or_prompt_to_batch_of_dict_conversations(
        messages_or_prompt:   Conversation
                            | list[Conversation]
                            | dict[Any, Conversation]
                            | str
                            | list[str]
                            | dict[any, Conversation]
    ) -> tuple[Batching, list[DictConversation]]:
        
        if isinstance(messages_or_prompt, str):
            return NoBatching(), [[{"role": "user", "content": messages_or_prompt}]]
        
        if isinstance(messages_or_prompt, dict):
            keys, values = unzip(messages_or_prompt.items())
            _, conversations = Chatbot._messages_or_prompt_to_batch_of_dict_conversations(values)
            return DictBatching(keys=keys), conversations

        # Note: This could be problematic because [] could be either a batch of type 0 or a
        # conversation with 0 messages. We hope that no one will ever needs conversations with 0
        # messages (and guess those are probably not supported by most APIs and libraries).
        if messages_or_prompt == []:
            return ListBatching(), []
        
        if messages_or_prompt == {}:
            return DictBatching(keys=[]), {}
        
        if isinstance(messages_or_prompt[0], str):
            for i, prompt in enumerate(messages_or_prompt):
                if not isinstance(prompt, str):
                    raise TypeError(
                        f"As the first element of `messages_of_prompt` is a string, all the other "
                        f"elements are expected to also be strings. But the element at index {i} "
                        f"is `{prompt}` of type `{type(prompt)}`."
                    )
            return ListBatching(), [ [{"role": "user", "content": prompt}]
                                     for prompt in messages_or_prompt ]
        
        if isinstance(messages_or_prompt[0], list):
            for i, conversation in enumerate(messages_or_prompt):
                if not isinstance(conversation, list):
                    raise TypeError(
                        f"As the first element of `messages_or_prompt` is a list, all other "
                        f"elements are expected to also be lists. But the element at index {i} "
                        f"is `{conversation}` of type `{type(conversation)}`."
                    )
            return ListBatching(), [ [Chatbot.to_dict_message(message) for message in conversation]
                                     for conversation in messages_or_prompt ]
        
        if isinstance(messages_or_prompt[0], (dict, tuple)):
            return NoBatching(), [[ Chatbot.to_dict_message(message)
                                    for message in messages_or_prompt ]]
    
        raise TypeError(
            f"`messages_or_prompt` should either be a string (one prompt), a list (list of "
            f"prompts, conversation, or list of conversations), or a dictionary (dictinoray whose "
            f"keys could be anything and whose values are conversations). But got "
            f"`{messages_or_prompt}` of type `{type(messages_or_prompt)}`."
        )

    # TO DO: to_tuple_message, to_dict_conversation, to_tuple_conversation
    @staticmethod
    def to_dict_message(message: Message) -> DictMessage:
        if isinstance(message, dict):
            if set(message.keys()) != {"role", "content"}:
                raise TypeError(
                    f"A message should be a dictionary with 'role' and 'content' keys, but the "
                    f"keys of the provided message dictionary are `{set(message.keys())}`."
                )
            if message["role"] not in ["user", "assistant", "system"]:
                raise TypeError(
                    f"The role of a message should be one of 'user', 'assistant', and 'system', "
                    f"but got `{message['role']}`."
                )
            return message
        
        if isinstance(message, tuple):
            if len(message) != 2:
                raise TypeError(
                    f"A message should be either a dict with 'role' and 'content' keys, but got "
                    f"`{message}` of length {len(message)}`."
                )
            role, content = message
            if not isinstance(role, Role):
                raise TypeError(
                    f"The role of a message should be `Role.user`, `Role.assistant`, or "
                    f"`Role.system`, but got `{role}` of type `{type(role)}`."
                )
            if not isinstance(content, str):
                raise TypeError(
                    f"The content of a message should be a string, but got `{content}` of type "
                    f"`{type(content)}`."
                )
            role_translations = { Role.user:      "user",
                                  Role.assistant: "assistant",
                                  Role.system:    "system" }
            role = role_translations[role]
            return {"role": role, "content": content}
        
        raise TypeError(
            f"A message should be either a tuple of length two containing the role and content or "
            f"a dictionary with keys 'role' and 'content', but got `{message}` of type `"
            f"`{type(message)}`."
        )

    @staticmethod
    def _validate_message_order(conversation: Conversation) -> None:
        if any(message["role"] == "system" for message in conversation[1:]):
            raise ValueError("Only the first message in a conversation can have the 'system' role.")
        
        if any( message["role"] == next_message["role"]
                for message, next_message in pairwise(conversation) ):
            raise ValueError(
                "Roles must alternate between 'user' and 'assistant' with eventually 'system' at "
                "the first position."
            )
        
@dataclass
class Dialogue:
    chatbot: Chatbot
    chatbot_role: Literal[Role.assistant, Role.user] = Role.assistant
    messages: TupleConversation = field(default_factory=[])

    def say(self, prompt: str) -> str:
        # to do: prefix cashing
        # speaker_role = {Role.user: Role.assistant, Role.assistant: Role.user}[]
        self.messages.append((self.chatbot_role.opposite(), prompt))
        response = self.chatbot(self.messages)
        self.messages.append(self.chatbot_role, response)
        return response
    
DebaterId = TypeVar("DebaterId")
def debate(
        max_messages: int,
        # to do: support any conversation, not just tuple conversations
        # to do: support giving a list of strings which will be passed as system messages
        # to do: first_messages is a bad name to mean any number of messages at the beginning of the converastion, find a better one
        first_messages: dict[DebaterId, TupleConversation] | list[TupleConversation],
        debater_ids: list[DebaterId] = [0, 1],
        chatbot: Chatbot | None = None,
        chatbots: dict[DebaterId, Chatbot] | list[Chatbot] | None = None,
        speaking_order: Iterable[DebaterId] = cycle([0, 1]),
        exit_condition_message: Callable[[DebaterId, str], bool] = (lambda _, __: False),
        exit_condition: Callable[[list[tuple[DebaterId, str]]], bool] = (lambda _: False),
        chatbot_kwargs: dict[str, Any] = {},
    ) -> list[tuple[DebaterId, str]]:
    
    if (chatbot is None) == (chatbots is None):
        raise ValueError("Either `chatbot` or `chatbots` should be provided, but not both.")

    if chatbots is None:
        chatbots = {id: chatbot for id in debater_ids}

    if isinstance(first_messages, list):
        if len(debater_ids) != len(first_messages):
            raise ValueError("`first_messages` should have the same length as `debater_ids`")
        first_messages = dict(zip(debater_ids, first_messages, strict=True))

    if set(first_messages.keys()) != set(debater_ids):
        raise ValueError("The keys of `first_messages` should be exactly `debater_ids`.")

    if isinstance(chatbots, list):
        if len(chatbots) != len(debater_ids):
            raise ValueError("`chatbots` should have the same length as `debater_ids`")
        chatbots = dict(zip(debater_ids, chatbots, strict=True))

    if set(chatbots.keys()) != set(debater_ids):
        raise ValueError("The keys of `chatbots` should be exactly `debater_ids`.")
    
    speaking_order = list(islice(speaking_order, max_messages))
    if not all(debater in debater_ids for debater in speaking_order):
        raise ValueError( "The elements of `speaking_order` should all be elements of "
                          "`debater_ids` or of `[0, 1]` if `debater_ids` is not provided." )

    generated_messages: list[DebaterId, str] = []
    for debater in speaking_order:
        # to do: make this work with chatbots which don't support consecutive user messages
        response = chatbots[debater](
                first_messages[debater]
            + [ ( (Role.assistant if debater_ == debater else Role.user),
                  content )
                for debater_, content in generated_messages ],
            **chatbot_kwargs
        )
        generated_messages.append((debater, response))

        if exit_condition_message(debater, response):
            break
        if exit_condition(generated_messages):
            break

    return generated_messages

def await_all(futures: list[asyncio.Future]) -> list:
    async def f():
        return await asyncio.gather(*futures)
    return asyncio.run(f())

class APIChatbot(Chatbot):
    is_fine_tuned: bool

    def _get_api_pricing(self) -> APIPricing | None:
        pricing_table = self._pricing_table()

        if pricing_table is None:
            return None
        
        matches = [ (model_name_regex, pricing)
                    for model_name_regex, pricing in pricing_table.items()
                    if re.fullmatch(model_name_regex, self._model_name) is not None ]
        
        if len(matches) == 0:
            print( "Could not estimate the price of the API call because the model name "
                   f"'{self._model_name}' is not in the the pricing table." )
            return None
        
        if len(matches) > 1:
            print( "Could not estimate the price of the API call because the model "
                   f"{self._model_name} matches more than one model name in the pricing table: "
                   + ", ".join( f"/{model_name_regex}/"
                                for model_name_regex in pricing_table.keys() ) )
            return None
        
        _, pricing = matches[0]

        return pricing

    def _estimate_completion_price_dollars( self,
                                             n_input_tokens: int,
                                             n_output_tokens: int,
                                             fine_tuned: bool,
                                             batch_api: bool ) -> float | None:
        
        pricing = self._get_api_pricing()

        if pricing is None:
            return None

        return pricing.completion_price_dollars( n_input_tokens  = n_input_tokens,
                                                 n_output_tokens = n_output_tokens,
                                                 fine_tuned      = fine_tuned,
                                                 batch_api       = batch_api )
    
    def _estimate_fine_tuning_price_dollars(self, n_training_tokens: int,
                                                  batch_api: bool ) -> float | None:
        
        pricing = self._get_api_pricing()

        if pricing is None:
            return False
        
        return pricing.fine_tuning_price_dollars( n_training_tokens = n_training_tokens,
                                                  batch_api         = batch_api )
    
    def _pricing_table(self) -> dict[str, APIPricing] | None:
        return None

    @abstractmethod
    async def _async_completion(self, messages: Conversation,
                                      max_new_tokens: int,
                                      temperature: float ) -> str:
        pass


    async def _completion_or_exception(self, messages: Conversation,
                                             max_new_tokens: int,
                                             temperature: float ) -> str | Exception:
        try:
            return await self._async_completion( messages    = messages,
                                                 max_new_tokens  = max_new_tokens,
                                                 temperature = temperature )
        except Exception as e:
            return e

    def _completion(self, conversations:  list[Conversation],
                          max_new_tokens: int,
                          temperature:    float,
                          print_price:    bool = True ) -> list[str]:

        completions = [None] * len(conversations)
        to_do_indices = list(range(len(conversations)))

        wait_until_retry_seconds = 1

        while True:
            new_completions = await_all(
                self._completion_or_exception( messages    = conversations[i],
                                               max_new_tokens  = max_new_tokens,
                                               temperature = temperature )
                for i in to_do_indices
            )
            
            for i, completion in zip(to_do_indices, new_completions):
                if not isinstance(completion, Exception):
                    completions[i] = completion

            to_do_indices = [ i
                              for i, completion in zip(to_do_indices, new_completions)
                              if isinstance(completion, Exception) ]
            
            exceptions = [ completion
                           for completion in new_completions
                           if isinstance(completion, Exception) ]
            
            if exceptions == []:
                break
            
            print(f"{len(exceptions)} API CALLS THREW AN EXCEPTION")
            print(f"THE FIRST EXCEPTION IS:", exceptions[0])
            if len(exceptions) > 1:
                print("THE OTHER API CALLS MAY OR MAY NOT HAVE THROWN THE SAME EXCEPTION")

            time.sleep(wait_until_retry_seconds)
            wait_until_retry_seconds *= 2

        if print_price:
            estimated_price = self._estimate_completion_price_dollars(
                n_input_tokens  = sum( len(message["content"])
                                       for conversation in conversations
                                       for message in conversation ),
                n_output_tokens = sum( len(completion)
                                       for completion in completions ),
                fine_tuned      = self.is_fine_tuned,
                batch_api       = False
            )        
            if estimated_price is not None:
                print(f"Estimated API call price: ${estimated_price:.02}")
            else:
                print("Could not estimate API call price.")

        return completions

@dataclass
class ChatbotAPIError(Exception):
    status_code: int
    message: str
    
    def __post_init__(self):
        super().__init__(f"Chatbot API responed with an error: {self.status_code} - {self.message}")

class TogetherAIChatbot(APIChatbot):
    _model_name:   str
    _api_key:      str
    is_fine_tuned: bool

    def __init__(self, model: str, api_key: str) -> None:
        self._model_name = model
        self._api_key = api_key
        self.is_fine_tuned = False
    
    async def _async_completion(self, messages: DictConversation,
                                      max_new_tokens: int,
                                      temperature: float ) -> str:
        
        url = "https://api.together.xyz/v1/chat/completions"
        headers = { "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json" }
        data = { "model": self._model_name,
                 "messages": messages,
                 "max_new_tokens": max_new_tokens,
                 "temperature": temperature }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response_json = await response.json()
            
        try:
            return response_json["choices"][0]["message"]["content"]
        
        except (KeyError, IndexError):
            if "error" in response_json:
                if "message" in response_json["error"]:
                    raise ChatbotAPIError( status_code = response.status,
                                           message     = str(response_json["error"]["message"]) )
                raise ChatbotAPIError( status_code = response.status,
                                       message     = str(response_json["error"]) )
            raise ChatbotAPIError( status_code = response.status,
                                   message     = str(response_json) )

last_called_mistral_api = 0

class MistralChatbot(APIChatbot):
    _model_name:   str
    _api_key:      str
    is_fine_tuned: bool

    def __init__(self, model: str, api_key: str) -> None:
        self._model_name = model
        self._api_key = api_key
        self.is_fine_tuned = False
    
    async def _async_completion(self, messages: DictConversation,
                                      max_new_tokens: int,
                                      temperature: float ) -> str:
        
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = { "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json" }
        data = { "model": self._model_name,
                 "messages": messages,
                 "max_tokens": max_new_tokens,
                 "temperature": temperature }
        
        global last_called_mistral_api

        # while perf_counter() <= last_called_mistral_api + 2.:
        #     await asyncio.sleep(0.1)

        last_called_mistral_api = perf_counter()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response_json = await response.json()
            
        try:
            return response_json["choices"][0]["message"]["content"]
        
        except (KeyError, IndexError):
            if "error" in response_json:
                if "message" in response_json["error"]:
                    raise ChatbotAPIError( status_code = response.status,
                                           message     = str(response_json["error"]["message"]) )
                raise ChatbotAPIError( status_code = response.status,
                                       message     = str(response_json["error"]) )
            raise ChatbotAPIError( status_code = response.status,
                                   message     = str(response_json) )

# TO DO: ACTUALLY IMPLEMENT THIS CLASS PROPERLY!!!!!!!!!!!!
class GoogleChatbot(APIChatbot):
    _model_name: str
    _model: google.generativeai.GenerativeModel

    async def _async_completion(self, messages: list[DictMessage],
                                      max_new_tokens: int,
                                      temperature: float ) -> str:
        
        # TO DO: FIX THIS
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        response = self._model.generate_content(
            messages[0]["content"],
            generation_config = google.generativeai.GenerationConfig(
                max_output_tokens = max_new_tokens,
                temperature = temperature
            )
        )
        # print(f"\n{hasattr(response, "text")=}")
        try:
            return response.text
        except ValueError:
            print("WARNING: GEMINI RETURNED SOME GOOFY ERROR")
            return "Sorry, I cannot assist with that."
    
    def __init__(self, model: str, api_key: str) -> None:
        google.generativeai.configure(api_key=api_key)
        self._model_name = model
        self._model = google.generativeai.GenerativeModel(model)
        self.is_fine_tuned = False

class AnthropicChatbot(APIChatbot):
    _model_name:   str
    _client:       AsyncAnthropic
    is_fine_tuned: bool

    def _pricing_table(self) -> APIPricing | None:
        return ANTHROPIC_API_PRICING

    def __init__(self, model: str, api_key: str) -> None:
        self._model_name = model
        self._client = Anthropic(api_key=api_key)
        self.is_fine_tuned = False

    async def _async_completion(self, messages: DictConversation,
                                      max_new_tokens: int,
                                      temperature: float) -> str:
        
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
            response = self._client.messages.create( system      = system_message,
                                                           messages    = messages,
                                                           max_tokens  = max_new_tokens,
                                                           model       = self._model_name,
                                                           temperature = temperature )
        
        else:
            response = self._client.messages.create( messages    = messages,
                                                           max_tokens  = max_new_tokens,
                                                           model       = self._model_name,
                                                           temperature = temperature )
        
        return response.content[0].text
    
class TemporaryOpenAIFile:
    client: OpenAI
    file_id: str

    def __init__(self, client: OpenAI, content: str, purpose: str) -> None:
        self.client = client
        file = self.client.files.create(file=BytesIO(content.encode()), purpose=purpose)
        self.file_id = file.id

    def __enter__(self) -> str:
        return self.file_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.files.delete(self.file_id)

class OpenAIFineTuningJob:
    client: OpenAI
    job_id: str

    def __init__(self, client: OpenAI,
                       training_file_id: str,
                       model: str,
                       hyperparameters: dict[str, Any] ) -> None:
        
        self.client = client
        print("CREATING OPENAI FINE-TUNING JOB")
        job = self.client.fine_tuning.jobs.create( training_file   = training_file_id,
                                                   model           = model,
                                                   hyperparameters = hyperparameters )
        self.job_id = job.id
        print("OPENAI FINE-TUNING JOB CREATED")

    def __enter__(self) -> str:
        return self.job_id

    def __exit__(self, exc_type, exc_val, ext_tb):
        try:
            job = self.client.fine_tuning.jobs.retrieve(self.job_id)
        except Exception as e:
            print("FEILED TO RETRIEVE OPENAI FINE TUNING JOB WITH ID", self.job_id)
            print("CANCEL THE JOB MANUALLY TO AVOID PAYING API CREDITS")
            print("EXCEPTION ENCOUNTERED WHILE TRYING TO RETRIEVE THE JOB:", e)
            raise e

        if job.status not in ["succeeded", "failed", "cancelled"]:
            print("CANCELING OPENAI FINE-TUNING JOB WITH ID", self.job_id)
            self.client.fine_tuning.jobs.cancel(self.job_id)

class OpenAIChatbot(APIChatbot):
    _model_name: str
    _api_key:    str
    is_fine_tuned: bool

    def _pricing_table(self) -> APIPricing | None:
        return OPENAI_API_PRICING

    def __init__(self, model: str, api_key: str) -> None:
        self._model_name = model
        self._api_key = api_key
        self.is_fine_tuned = False

    async def _async_completion(self, messages:    Conversation,
                                      max_new_tokens:  int,
                                      temperature: float ) -> str:

        # we don't use the openai library because it doesn't support async

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        data = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response = await response.json()
            
        # should i like handle what happens if response is an error here?

        return response["choices"][0]["message"]["content"]
    
    def fine_tune(
        self,
        conversations:            list[Conversation],
        batch_size:               int | Literal["auto"] = "auto",
        learning_rate_multiplier: int | Literal["auto"] = "auto",
        n_epochs:                 int | Literal["auto"] = "auto",
        print_price:              bool = True
    ) -> "OpenAIChatbot":

        client = OpenAI(api_key=self._api_key)

        _, conversations = self._messages_or_prompt_to_batch_of_dict_conversations(conversations)
        fine_tuning_data = "\n".join( json.dumps({"messages": conversation})
                                      for conversation in conversations )
        hyperparameters = { "batch_size":               batch_size,
                            "learning_rate_multiplier": learning_rate_multiplier,
                            "n_epochs":                 n_epochs }
        with TemporaryOpenAIFile(client, content=fine_tuning_data, purpose="fine-tune") as file_id:
            with OpenAIFineTuningJob( client,
                                      file_id,
                                      self._model_name,
                                      hyperparameters ) as fine_tuning_job_id:
                printed_messages = set()
                for i in count():
                    time.sleep(1)

                    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id)
                    for event in reversed(list(events)):
                        if event.message not in printed_messages:
                            print(event.message)
                            printed_messages.add(event.message)
                    
                    retrieved_job = client.fine_tuning.jobs.retrieve(fine_tuning_job_id)

                    estimated_finish = retrieved_job.estimated_finish
                    if i % 10 == 0 and estimated_finish is not None:
                        print( "Estimated remaining time:",
                               datetime.fromtimestamp(estimated_finish) - datetime.now() )

                    if retrieved_job.status == "succeeded":
                        break
                    
                    if retrieved_job.status == "failed":
                        print( "OPENAI FINE-TUNING JOB FAILED. ERROR MESSAGE:",
                               retrieved_job.error.message )
                        raise ValueError(   "OpenAI fine-tuning job failed with message: "
                                          + retrieved_job.error.message )
                    
                    if retrieved_job.status == "cancelled":
                        print("OPENAI FINE-TUNING JOB WAS CANCELLED UNEXPECTEDLY")
                        raise ValueError("OpenAI fine-tuning job was cancelled unexpectedly.")
                    
        if print_price:
            estimated_price = self._estimate_fine_tuning_price_dollars(
                n_training_tokens = sum( len(content)
                                         for conversation in conversations
                                         for role, content in conversation ),
                batch_api         = False
            )
            if estimated_price is None:
                print("Could not estimate fine tuning API price.")
            else:
                print(f"Estimated fine tuning API price: ${estimated_price:.02}")

        fine_tuned_chatbot = deepcopy(self)
        fine_tuned_chatbot._model_name = retrieved_job.fine_tuned_model
        fine_tuned_chatbot.is_fine_tuned = True
        return fine_tuned_chatbot

X = TypeVar("X")
Y = TypeVar("Y")
def unzip(itr: Iterable[tuple[X, Y]]) -> list[tuple[X, Y]]:
    xs = []
    ys = []
    for x, y in itr:
        xs.append(x)
        ys.append(y)
    return xs, ys
