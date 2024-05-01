# # Q-1
total_cards= 52
total_suits= 4
total_hearts= 13
total_king = 4
king_of_hearts = 1
print("Probability of drawing a heart: ", total_hearts/total_cards)
print("Probability of drawing a king: ", total_king/total_cards)
print("Probability of drawing a king of hearts: ",king_of_hearts/total_hearts)

# # Q-2
prob_spam = 0.2
prob_not_spam = 0.8
prob_detected_spam_given_spam=0.95
prob_not_detected_spam_given_not_spam=0.98
prob_not_detected_spam_given_spam = 0.02
prob_detected_spam = prob_detected_spam_given_spam*prob_spam + prob_not_detected_spam_given_spam*prob_not_spam
prob_spam_given_detected_spam = (prob_detected_spam_given_spam*prob_spam)/prob_detected_spam
print(f"{prob_spam_given_detected_spam:.2f}")

# # Q-3
prob_disease= 0.02
prob_positive = 0.98
prob_not_disease = 1-prob_disease
prob_negative = 0.90
prob_positive_have_disease = prob_disease * prob_positive
p_pos = (prob_disease * prob_positive)+(prob_not_disease * (1-prob_negative))
prob_disease_given_positive = prob_positive_have_disease/p_pos
print("Probab that a person who is tested positive actually have the disease is - ",prob_disease_given_positive)
