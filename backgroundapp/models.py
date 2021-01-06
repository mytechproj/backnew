from django.db import models


# Makemigrations - Save changes in a file
# Migrate - Apply the changes to the db
# Create your models here.
class Contact(models.Model):
    name = models.CharField(max_length=122)
    email = models.CharField(max_length=122)
    desc = models.TextField(default="0")
    date = models.DateField()

    def __str__(self):
        return self.name
    