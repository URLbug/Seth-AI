from aiogram import Bot, Dispatcher, executor, types

from new import plan_b


bot = Bot(token='5619500244:AAEnja91bgcnolbKxImzGi15mpqTCGpmUq0')
dp = Dispatcher(bot)

@dp.message_handler()
async def start(m: types.Message):
  v = ''
  v = plan_b(m.text)
  await m.reply(v)

if __name__ == '__main__':
  executor.start_polling(dp, skip_updates=True)
  