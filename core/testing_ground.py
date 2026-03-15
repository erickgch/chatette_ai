from note_manager import delete_reminder_by_line, get_all_reminders

print("Before:")
print(get_all_reminders())

result = delete_reminder_by_line("[2026-03-11 20:41] Dinner with Ines and Annika on March 12th, at 18h.")
print(f"\nResult: {result}")

print("\nAfter:")
print(get_all_reminders())