from __future__ import unicode_literals

from django.db import models


class Client(models.Model):
    """
    """
    name = models.CharField(max_length=75, db_index=True)
    public_key = models.CharField(max_length=16, db_index=True)
    training = models.BooleanField()


class Customer(models.Model):
    """
    """
    client = models.ForeignKey('Client')
    identifier = models.CharField(max_length=14, db_index=True)
    name = models.CharField(max_length=100, db_index=True)


class Transaction(models.Model):
    """
    """
    client = models.ForeignKey('Client')
    customer = models.ForeignKey('Customer')
    value = models.PositiveIntegerField()
