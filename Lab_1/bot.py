from sopel import module
from emo.wdemotions import EmotionDetector

emo = EmotionDetector()

@module.rule('')
def hi(bot, trigger):
    print(trigger, trigger.nick)
    emotions = emo.detect_emotion_in_raw(trigger)
    emotions = emotions.tolist()
    bot.say(str(emotions))
