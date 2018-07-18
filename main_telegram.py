import sys
import time
import telepot
import telepot.helper
from telepot.loop import MessageLoop
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
from telepot.delegate import (per_chat_id, create_open, pave_event_space, include_callback_query_chat_id)
import datetime
import telegram
from telepot.namedtuple import InlineQueryResultArticle, InlineQueryResultPhoto
from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
from telepot.namedtuple import InlineQueryResultArticle, InputTextMessageContent, \
		InlineQueryResultPhoto
import emoji

now = datetime.datetime.now()
propose_records = telepot.helper.SafeDict()


import wikipedia
from googletrans import Translator
translator = Translator()


	
class Bebot(telepot.helper.ChatHandler):

	#variables
	keyboard = InlineKeyboardMarkup(inline_keyboard=[[
					InlineKeyboardButton(text='Medicina', callback_data='Medicina'),
					InlineKeyboardButton(text='Voli', callback_data='Voli')]])
	MESSAGGIO = "Sono Bebot, il chatbot di Be! \n Posso aiutarti a chiarire dubbi nelle aree di Medicina Cellulare e Diritti dei Passeggeri sui Voli.\n Scegli una delle seguenti aree:"
	USER = ""
	HOUR = now.hour
	dis = 0
	
	
	def __init__(self, *args, **kwargs):
		super(Bebot, self).__init__(*args, **kwargs)
		# Retrieve from database
		global propose_records
		if self.id in propose_records:
			self._count, self._edit_msg_ident = propose_records[self.id]
			self._editor = telepot.helper.Editor(self.bot, self._edit_msg_ident) if self._edit_msg_ident else None
		else:
			self._count = 0
			self._edit_msg_ident = None
			self._editor = None
		self._greetings()
		
	def _greetings(self):
		sent = self.sender.sendMessage( str(self.greetings_words()) + ' ' + str(self.USER) + '!\n' + self.MESSAGGIO + "\n")
		sent = self.sender.sendPhoto(open('C:\\Users\\l.gomes\\Pictures\\Saved Pictures\\medicin.jpg', 'rb'), reply_markup=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='Medicina', callback_data='Medicina', url = 'https://telegram.me/Be_med_bot')]]))
		sent = self.sender.sendPhoto(open('C:\\Users\\l.gomes\\Pictures\\Saved Pictures\\airflight.jpg', 'rb'), reply_markup=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='Voli', callback_data='Voli', url = 'https://telegram.me/be_chatbot')]]))
		
		
	#choose vocabulary according with time
	def greetings_words(self):
		if 4 <= self.HOUR < 14:
			greet = "Buongiorno"
		else:
			greet = "Buonasera"
		return greet
	
	#cancel last message
	def _cancel_last(self):
		if self._editor:
			self._editor.editMessageReplyMarkup(reply_markup=None)
			self._editor = None
			self._edit_msg_ident = None
			
		
	def on_chat_message(self, msg):
		
		print (str(msg))
		#in caso sia out of scope
		disambiguation = ["Hmm... penso che non ho capito la tua domanda :sweat_smile:. \n Puoi essere più preciso nella tua richiesta?",
							"Mi dispiace, non so risponderti ma metterò in considerazione la domanda in futuro. :confused: ",
							"Non ho capito quello che hai scritto nuovamente :sob: Puoi riformulare la tua domanda? "]
		
		input_question = translator.translate(msg['text'], dest='en').text
		try:
			page = wikipedia.summary(input_question)
			fr = translator.translate(page, dest='it')
			print(input_question)
			sent = self.sender.sendMessage(fr.text)
			return;
		except wikipedia.exceptions.DisambiguationError as e:
			sent = self.sender.sendMessage(emoji.emojize(disambiguation[self.dis], use_aliases=True))
			self.dis = self.dis + 1
			if self.dis > 2: 
				self.dis = 0
			return;
		except:
			sent = self.sender.sendMessage(emoji.emojize("Non so di cosa si tratta. :confused: ", use_aliases=True))
		
		# ny = wikipedia.page(input_question)
		# sent = self.sender.sendMessage(ny.url)

		#self.USER = (str(msg["from"]["first_name"]))

		# #print (msg["text"])
		# #frasi composte separate da "_"
		#sent = self.sender.sendMessage('https://en.wikipedia.org/wiki/' + str(msg["text"])) 
	
	def on_callback_query(self, msg):
		
		
		query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
		
		#save data in db according with user's feedback
		if query_data == 'aiuto_si':
			self._cancel_last()
			self.sender.sendMessage('Sono contento di aver aiutato!')
			self.close()
			
		elif query_data == 'aiuto_no':
			self._cancel_last()
			self.sender.sendMessage('Mi dispiace. Puoi provare a riformulare la domanda o contattare il seguente recapito: XXX.')
			self.close()
		
				
	# def on__idle(self, event):
		# self._count += 1
		# sent = self.sender.sendMessage('Ti ho aiutato?',reply_markup=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='Si', callback_data='aiuto_si'),InlineKeyboardButton(text='No', callback_data='aiuto_no')]]))
		# self._editor = telepot.helper.Editor(self.bot, sent)
		# self._edit_msg_ident = telepot.message_identifier(sent)
		
	# def on_close(self, ex):
		# global propose_records
		# propose_records[self.id] = (self._count, self._edit_msg_ident)
		
		
TOKEN = '588907997:AAHDaBq3DEpTyqEfJaSqdQTJw4-xgLO4Aaw'
TIMEOUT = 1000

bot = telepot.DelegatorBot(TOKEN, [
    include_callback_query_chat_id(
        pave_event_space())(
            per_chat_id(types=['private']), create_open, Bebot, timeout=TIMEOUT),
])
MessageLoop(bot).run_as_thread()
print('Listening ...')

while 1:
	time.sleep(TIMEOUT)