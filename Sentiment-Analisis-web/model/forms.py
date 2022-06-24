from django import forms

class textForm(forms.Form):
    text = forms.CharField(label='text', max_length=100)