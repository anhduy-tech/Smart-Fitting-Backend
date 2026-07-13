from django.apps import apps
from django.db.models import BooleanField
from rest_framework import serializers
from app.models import *


def get_serializer(name):
    """
    Generic serializer factory - tạo serializer động dựa trên tên model.
    Tuân theo pattern từ views_ex.py
    """
    try:
        Model = apps.get_model('app', name)
    except:
        return None, None

    # QUAN TRONG: DRF KHONG tu dong ke thua default=True (hay bat ky
    # default nao) tu model sang serializer field cho BooleanField. Khi
    # request gui qua form-data/multipart va KHONG gui field do, DRF coi
    # nhu 1 checkbox HTML chua duoc tick -> tu dong gan False, BAT KE
    # model co default=True hay khong (vd Product.is_active, Frame.is_active).
    # Phai chi dinh ro extra_kwargs['default'] cho tung BooleanField de
    # serializer dung dung gia tri mac dinh cua model khi field bi thieu.
    boolean_field_kwargs = {
        f.name: {'default': f.default}
        for f in Model._meta.fields
        if isinstance(f, BooleanField) and f.has_default()
    }

    class GenericSerializer(serializers.ModelSerializer):
        class Meta:
            model = Model
            fields = '__all__'
            read_only_fields = ['created_at', 'updated_at'] if hasattr(Model, 'created_at') else []
            extra_kwargs = boolean_field_kwargs

        def create(self, validated_data):
            return Model.objects.create(**validated_data)
        
        def update(self, instance, validated_data):
            for attr, value in validated_data.items():
                setattr(instance, attr, value)
            instance.save()
            return instance

    return Model, GenericSerializer


# =============================================================================
# Specific Serializers (cho các trường hợp đặc biệt)
# =============================================================================

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'phone', 'full_name', 'email', 'date_of_birth', 
                  'address', 'avatar', 'gender', 'height', 'weight', 
                  'role', 'is_verified', 'created_at']
        read_only_fields = ['created_at', 'role', 'is_verified']


class UserRegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = ['phone', 'password', 'full_name', 'email']

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User.objects.create(**validated_data)
        user.set_password(password)
        user.save()
        return user


class OTPSerializer(serializers.ModelSerializer):
    class Meta:
        model = OTP
        fields = '__all__'


class LoginSerializer(serializers.Serializer):
    phone = serializers.CharField(max_length=15)
    password = serializers.CharField(write_only=True)


class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True, min_length=6)


class ProductSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    
    class Meta:
        model = Product
        fields = '__all__'


class OrderSerializer(serializers.ModelSerializer):
    items = serializers.SerializerMethodField()

    class Meta:
        model = Order
        fields = '__all__'
        read_only_fields = ['order_code', 'created_at', 'updated_at']

    def get_items(self, obj):
        items = Order_Item.objects.filter(order=obj)
        return OrderItemSerializer(items, many=True).data


class OrderItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Order_Item
        fields = '__all__'


class SupportTicketSerializer(serializers.ModelSerializer):
    replies = serializers.SerializerMethodField()

    class Meta:
        model = Support_Ticket
        fields = '__all__'
        read_only_fields = ['ticket_code', 'created_at', 'updated_at']

    def get_replies(self, obj):
        replies = Support_Reply.objects.filter(ticket=obj)
        return SupportReplySerializer(replies, many=True).data


class SupportReplySerializer(serializers.ModelSerializer):
    class Meta:
        model = Support_Reply
        fields = '__all__'


class GeneratedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Generated_Image
        fields = '__all__'
        read_only_fields = ['created_at']


class CartSerializer(serializers.ModelSerializer):
    product_name = serializers.CharField(source='product.name', read_only=True)
    product_image = serializers.ImageField(source='product.image', read_only=True)
    product_price = serializers.DecimalField(source='product.price', max_digits=12, decimal_places=2, read_only=True)

    class Meta:
        model = Cart
        fields = ['id', 'user', 'product', 'product_name', 'product_image', 
                  'product_price', 'size', 'quantity', 'created_at']


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = '__all__'