import os
from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.utils import timezone
from datetime import date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_folder = os.path.join(BASE_DIR, "static")


# =============================================================================
# Custom User Manager
# =============================================================================
class UserManager(BaseUserManager):
    def create_user(self, phone, password=None, **extra_fields):
        if not phone:
            raise ValueError('Số điện thoại là bắt buộc')
        user = self.model(phone=phone, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, phone, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        extra_fields.setdefault('role', 'admin')
        return self.create_user(phone, password, **extra_fields)


# =============================================================================
# 1.1.1 User - Người dùng (xác thực bằng số điện thoại)
# =============================================================================
class User(AbstractUser):
    username = None  # Không dùng username
    phone = models.CharField(max_length=15, unique=True, verbose_name='Số điện thoại')
    email = models.EmailField(max_length=255, blank=True, null=True, verbose_name='Email')
    full_name = models.CharField(max_length=255, blank=True, null=True, verbose_name='Họ tên')
    date_of_birth = models.DateField(blank=True, null=True, verbose_name='Ngày sinh')
    address = models.TextField(blank=True, null=True, verbose_name='Địa chỉ')
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True, verbose_name='Ảnh đại diện')
    role = models.CharField(max_length=20, choices=[
        ('user', 'Người dùng'),
        ('admin', 'Quản trị viên'),
    ], default='user', verbose_name='Vai trò')
    gender = models.CharField(max_length=10, choices=[
        ('male', 'Nam'),
        ('female', 'Nữ'),
        ('other', 'Khác'),
    ], blank=True, null=True, verbose_name='Giới tính')
    height = models.FloatField(blank=True, null=True, verbose_name='Chiều cao (cm)')
    weight = models.FloatField(blank=True, null=True, verbose_name='Cân nặng (kg)')
    is_verified = models.BooleanField(default=False, verbose_name='Đã xác thực')
    device_token = models.CharField(max_length=500, blank=True, null=True, verbose_name='Device Token')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    USERNAME_FIELD = 'phone'
    REQUIRED_FIELDS = []

    objects = UserManager()

    class Meta:
        db_table = 'user'
        verbose_name = 'Người dùng'
        verbose_name_plural = 'Người dùng'
        ordering = ['-created_at']

    def __str__(self):
        return self.full_name or self.phone


# =============================================================================
# OTP - Mã xác thực
# =============================================================================
class OTP(models.Model):
    phone = models.CharField(max_length=15, verbose_name='Số điện thoại')
    code = models.CharField(max_length=6, verbose_name='Mã OTP')
    purpose = models.CharField(max_length=20, choices=[
        ('register', 'Đăng ký'),
        ('forgot_password', 'Quên mật khẩu'),
        ('login', 'Đăng nhập'),
    ], default='register', verbose_name='Mục đích')
    is_used = models.BooleanField(default=False, verbose_name='Đã sử dụng')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')
    expires_at = models.DateTimeField(verbose_name='Hết hạn')
    ip_address = models.GenericIPAddressField(blank=True, null=True, verbose_name='Địa chỉ IP')

    class Meta:
        db_table = 'otp'
        verbose_name = 'Mã OTP'
        verbose_name_plural = 'Mã OTP'
        ordering = ['-created_at']

    def is_expired(self):
        return timezone.now() > self.expires_at

    def __str__(self):
        return f'{self.phone} - {self.code}'


# =============================================================================
# Product_Category - Danh mục sản phẩm (Quần / Áo / Váy)
# =============================================================================
class Product_Category(models.Model):
    name = models.CharField(max_length=255, verbose_name='Tên danh mục')
    code = models.CharField(max_length=50, unique=True, verbose_name='Mã danh mục')
    description = models.TextField(blank=True, null=True, verbose_name='Mô tả')
    category_type = models.CharField(max_length=20, choices=[
        ('shirt', 'Áo'),
        ('pants', 'Quần'),
        ('dress', 'Váy'),
    ], verbose_name='Loại')
    image = models.ImageField(upload_to='categories/', blank=True, null=True, verbose_name='Ảnh danh mục')
    sort_order = models.IntegerField(default=0, verbose_name='Thứ tự')
    is_active = models.BooleanField(default=True, verbose_name='Kích hoạt')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    class Meta:
        db_table = 'product_category'
        verbose_name = 'Danh mục sản phẩm'
        verbose_name_plural = 'Danh mục sản phẩm'
        ordering = ['sort_order', 'name']

    def __str__(self):
        return self.name


# =============================================================================
# Product - Sản phẩm
# =============================================================================
class Product(models.Model):
    category = models.ForeignKey(Product_Category, on_delete=models.CASCADE, related_name='products', verbose_name='Danh mục')
    name = models.CharField(max_length=255, verbose_name='Tên sản phẩm')
    code = models.CharField(max_length=50, unique=True, verbose_name='Mã sản phẩm')
    description = models.TextField(blank=True, null=True, verbose_name='Mô tả')
    material = models.CharField(max_length=255, blank=True, null=True, verbose_name='Chất liệu')
    color = models.CharField(max_length=100, blank=True, null=True, verbose_name='Màu sắc')
    size = models.CharField(max_length=50, verbose_name='Kích cỡ')
    price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name='Giá bán')
    original_price = models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True, verbose_name='Giá gốc')
    image = models.ImageField(upload_to='products/', verbose_name='Ảnh sản phẩm')
    images = models.JSONField(blank=True, null=True, verbose_name='Danh sách ảnh')
    stock = models.IntegerField(default=0, verbose_name='Tồn kho')
    is_active = models.BooleanField(default=True, verbose_name='Kích hoạt')
    sort_order = models.IntegerField(default=0, verbose_name='Thứ tự')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    class Meta:
        db_table = 'product'
        verbose_name = 'Sản phẩm'
        verbose_name_plural = 'Sản phẩm'
        ordering = ['sort_order', '-created_at']

    def __str__(self):
        return f'{self.name} - {self.size}'


# =============================================================================
# Frame_Category - Danh mục khung nền
# =============================================================================
class Frame_Category(models.Model):
    name = models.CharField(max_length=255, verbose_name='Tên danh mục')
    code = models.CharField(max_length=50, unique=True, verbose_name='Mã danh mục')
    description = models.TextField(blank=True, null=True, verbose_name='Mô tả')
    image = models.ImageField(upload_to='frame_categories/', blank=True, null=True, verbose_name='Ảnh mẫu')
    sort_order = models.IntegerField(default=0, verbose_name='Thứ tự')
    is_active = models.BooleanField(default=True, verbose_name='Kích hoạt')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    class Meta:
        db_table = 'frame_category'
        verbose_name = 'Danh mục khung nền'
        verbose_name_plural = 'Danh mục khung nền'
        ordering = ['sort_order', 'name']

    def __str__(self):
        return self.name


# =============================================================================
# Frame - Khung nền
# =============================================================================
class Frame(models.Model):
    category = models.ForeignKey(Frame_Category, on_delete=models.CASCADE, related_name='frames', verbose_name='Danh mục')
    name = models.CharField(max_length=255, verbose_name='Tên khung nền')
    code = models.CharField(max_length=50, unique=True, verbose_name='Mã khung nền')
    image = models.ImageField(upload_to='frames/', verbose_name='Ảnh nền')
    description = models.TextField(blank=True, null=True, verbose_name='Mô tả')
    tags = models.CharField(max_length=500, blank=True, null=True, verbose_name='Tags (phân cách bởi dấu phẩy)')
    is_active = models.BooleanField(default=True, verbose_name='Kích hoạt')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    class Meta:
        db_table = 'frame'
        verbose_name = 'Khung nền'
        verbose_name_plural = 'Khung nền'
        ordering = ['-created_at']

    def __str__(self):
        return self.name


# =============================================================================
# Portrait_Photo - Ảnh chân dung người dùng
# =============================================================================
class Portrait_Photo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='portrait_photos', verbose_name='Người dùng')
    original_image = models.ImageField(upload_to='portraits/original/', verbose_name='Ảnh gốc')
    processed_image = models.ImageField(upload_to='portraits/processed/', blank=True, null=True, verbose_name='Ảnh đã xử lý (tách nền)')
    has_background_removed = models.BooleanField(default=False, verbose_name='Đã tách nền')
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Chờ xử lý'),
        ('processing', 'Đang xử lý'),
        ('completed', 'Hoàn thành'),
        ('failed', 'Thất bại'),
        ('rejected', 'Từ chối'),
    ], default='pending', verbose_name='Trạng thái')
    nsfw_check = models.BooleanField(default=False, verbose_name='Vi phạm NSFW')
    celebrity_check = models.BooleanField(default=False, verbose_name='Người nổi tiếng')
    reject_reason = models.CharField(max_length=255, blank=True, null=True, verbose_name='Lý do từ chối')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tải lên')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    class Meta:
        db_table = 'portrait_photo'
        verbose_name = 'Ảnh chân dung'
        verbose_name_plural = 'Ảnh chân dung'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.phone} - {self.status}'


# =============================================================================
# Generated_Image - Ảnh sinh ra từ hệ thống thử đồ
# =============================================================================
class Generated_Image(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='generated_images', verbose_name='Người dùng')
    portrait = models.ForeignKey(Portrait_Photo, on_delete=models.SET_NULL, null=True, blank=True, related_name='generated_images', verbose_name='Ảnh chân dung')
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True, related_name='generated_images', verbose_name='Sản phẩm')
    frame = models.ForeignKey(Frame, on_delete=models.SET_NULL, null=True, blank=True, related_name='generated_images', verbose_name='Khung nền')
    result_image = models.ImageField(upload_to='generated/', verbose_name='Ảnh kết quả')
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Chờ xử lý'),
        ('processing', 'Đang xử lý'),
        ('completed', 'Hoàn thành'),
        ('failed', 'Thất bại'),
    ], default='pending', verbose_name='Trạng thái')
    processing_time = models.FloatField(blank=True, null=True, verbose_name='Thời gian xử lý (giây)')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')

    class Meta:
        db_table = 'generated_image'
        verbose_name = 'Ảnh đã sinh'
        verbose_name_plural = 'Ảnh đã sinh'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.phone} - {self.product.name if self.product else "N/A"}'


# =============================================================================
# Order - Đơn hàng
# =============================================================================
class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders', verbose_name='Người dùng')
    order_code = models.CharField(max_length=50, unique=True, verbose_name='Mã đơn hàng')
    full_name = models.CharField(max_length=255, verbose_name='Người nhận')
    phone = models.CharField(max_length=15, verbose_name='SĐT người nhận')
    address = models.TextField(verbose_name='Địa chỉ giao hàng')
    note = models.TextField(blank=True, null=True, verbose_name='Ghi chú')
    subtotal = models.DecimalField(max_digits=12, decimal_places=2, verbose_name='Tạm tính')
    shipping_fee = models.DecimalField(max_digits=12, decimal_places=2, default=0, verbose_name='Phí vận chuyển')
    discount = models.DecimalField(max_digits=12, decimal_places=2, default=0, verbose_name='Giảm giá')
    total = models.DecimalField(max_digits=12, decimal_places=2, verbose_name='Tổng cộng')
    payment_method = models.CharField(max_length=20, choices=[
        ('cod', 'COD'),
        ('qr', 'QR Code'),
    ], default='cod', verbose_name='Phương thức thanh toán')
    payment_status = models.CharField(max_length=20, choices=[
        ('pending', 'Chưa thanh toán'),
        ('paid', 'Đã thanh toán'),
        ('failed', 'Thất bại'),
        ('refunded', 'Đã hoàn tiền'),
    ], default='pending', verbose_name='Trạng thái thanh toán')
    payment_qr = models.ImageField(upload_to='payments/', blank=True, null=True, verbose_name='Mã QR thanh toán')
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Chờ xác nhận'),
        ('confirmed', 'Đã xác nhận'),
        ('shipping', 'Đang giao'),
        ('delivered', 'Đã giao'),
        ('cancelled', 'Đã hủy'),
        ('returned', 'Trả hàng'),
    ], default='pending', verbose_name='Trạng thái đơn hàng')
    tracking_number = models.CharField(max_length=100, blank=True, null=True, verbose_name='Mã vận đơn')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    class Meta:
        db_table = 'order'
        verbose_name = 'Đơn hàng'
        verbose_name_plural = 'Đơn hàng'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.order_code} - {self.user.phone}'


# =============================================================================
# Order_Item - Chi tiết đơn hàng
# =============================================================================
class Order_Item(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items', verbose_name='Đơn hàng')
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, related_name='order_items', verbose_name='Sản phẩm')
    product_name = models.CharField(max_length=255, verbose_name='Tên sản phẩm')
    product_code = models.CharField(max_length=50, verbose_name='Mã sản phẩm')
    product_image = models.CharField(max_length=500, blank=True, null=True, verbose_name='Ảnh sản phẩm')
    size = models.CharField(max_length=50, verbose_name='Kích cỡ')
    quantity = models.IntegerField(default=1, verbose_name='Số lượng')
    price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name='Đơn giá')
    subtotal = models.DecimalField(max_digits=12, decimal_places=2, verbose_name='Thành tiền')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')

    class Meta:
        db_table = 'order_item'
        verbose_name = 'Chi tiết đơn hàng'
        verbose_name_plural = 'Chi tiết đơn hàng'

    def __str__(self):
        return f'{self.order.order_code} - {self.product_name}'


# =============================================================================
# Support_Ticket - Yêu cầu hỗ trợ / Khiếu nại
# =============================================================================
class Support_Ticket(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='support_tickets', verbose_name='Người dùng')
    ticket_code = models.CharField(max_length=50, unique=True, verbose_name='Mã yêu cầu')
    subject = models.CharField(max_length=255, verbose_name='Tiêu đề')
    message = models.TextField(verbose_name='Nội dung')
    category = models.CharField(max_length=20, choices=[
        ('support', 'Hỗ trợ'),
        ('complaint', 'Khiếu nại'),
        ('feedback', 'Góp ý'),
    ], default='support', verbose_name='Loại')
    priority = models.CharField(max_length=20, choices=[
        ('low', 'Thấp'),
        ('medium', 'Trung bình'),
        ('high', 'Cao'),
    ], default='medium', verbose_name='Ưu tiên')
    status = models.CharField(max_length=20, choices=[
        ('open', 'Mở'),
        ('processing', 'Đang xử lý'),
        ('resolved', 'Đã giải quyết'),
        ('closed', 'Đã đóng'),
    ], default='open', verbose_name='Trạng thái')
    assigned_to = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='assigned_tickets', verbose_name='Người xử lý')
    images = models.JSONField(blank=True, null=True, verbose_name='Ảnh đính kèm')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    class Meta:
        db_table = 'support_ticket'
        verbose_name = 'Yêu cầu hỗ trợ'
        verbose_name_plural = 'Yêu cầu hỗ trợ'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.ticket_code} - {self.subject}'


# =============================================================================
# Support_Reply - Phản hồi yêu cầu hỗ trợ
# =============================================================================
class Support_Reply(models.Model):
    ticket = models.ForeignKey(Support_Ticket, on_delete=models.CASCADE, related_name='replies', verbose_name='Yêu cầu')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='support_replies', verbose_name='Người phản hồi')
    message = models.TextField(verbose_name='Nội dung')
    images = models.JSONField(blank=True, null=True, verbose_name='Ảnh đính kèm')
    is_admin = models.BooleanField(default=False, verbose_name='Admin phản hồi')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')

    class Meta:
        db_table = 'support_reply'
        verbose_name = 'Phản hồi hỗ trợ'
        verbose_name_plural = 'Phản hồi hỗ trợ'
        ordering = ['created_at']

    def __str__(self):
        return f'{self.ticket.ticket_code} - {self.user.phone}'


# =============================================================================
# Uploaded_Image - Ảnh tải lên (kiểm tra NSFW / Celebrity)
# =============================================================================
class Uploaded_Image(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_images', verbose_name='Người dùng')
    image = models.ImageField(upload_to='uploads/', verbose_name='Ảnh')
    image_type = models.CharField(max_length=20, choices=[
        ('portrait', 'Chân dung'),
        ('product', 'Sản phẩm'),
        ('frame', 'Khung nền'),
        ('other', 'Khác'),
    ], default='other', verbose_name='Loại ảnh')
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Chờ kiểm tra'),
        ('approved', 'Đã duyệt'),
        ('rejected', 'Từ chối'),
    ], default='pending', verbose_name='Trạng thái')
    nsfw_score = models.FloatField(blank=True, null=True, verbose_name='Điểm NSFW')
    is_nsfw = models.BooleanField(default=False, verbose_name='Vi phạm NSFW')
    is_celebrity = models.BooleanField(default=False, verbose_name='Người nổi tiếng')
    celebrity_name = models.CharField(max_length=255, blank=True, null=True, verbose_name='Tên người nổi tiếng')
    auto_deleted = models.BooleanField(default=False, verbose_name='Đã tự động xóa')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tải lên')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    class Meta:
        db_table = 'uploaded_image'
        verbose_name = 'Ảnh tải lên'
        verbose_name_plural = 'Ảnh tải lên'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.phone} - {self.image_type} - {self.status}'


# =============================================================================
# Notification - Thông báo push
# =============================================================================
class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications', verbose_name='Người dùng')
    title = models.CharField(max_length=255, verbose_name='Tiêu đề')
    body = models.TextField(verbose_name='Nội dung')
    notification_type = models.CharField(max_length=50, choices=[
        ('order', 'Đơn hàng'),
        ('promotion', 'Khuyến mãi'),
        ('system', 'Hệ thống'),
        ('support', 'Hỗ trợ'),
    ], default='system', verbose_name='Loại thông báo')
    data = models.JSONField(blank=True, null=True, verbose_name='Dữ liệu đính kèm')
    is_read = models.BooleanField(default=False, verbose_name='Đã đọc')
    sent_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày gửi')
    read_at = models.DateTimeField(blank=True, null=True, verbose_name='Ngày đọc')

    class Meta:
        db_table = 'notification'
        verbose_name = 'Thông báo'
        verbose_name_plural = 'Thông báo'
        ordering = ['-sent_at']

    def __str__(self):
        return f'{self.user.phone} - {self.title}'


# =============================================================================
# Cart - Giỏ hàng (lưu tạm trước khi đặt hàng)
# =============================================================================
class Cart(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='cart_items', verbose_name='Người dùng')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='cart_items', verbose_name='Sản phẩm')
    size = models.CharField(max_length=50, verbose_name='Kích cỡ')
    quantity = models.IntegerField(default=1, verbose_name='Số lượng')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày thêm')

    class Meta:
        db_table = 'cart'
        verbose_name = 'Giỏ hàng'
        verbose_name_plural = 'Giỏ hàng'
        unique_together = ('user', 'product', 'size')

    def __str__(self):
        return f'{self.user.phone} - {self.product.name} x{self.quantity}'


# =============================================================================
# Favourite - Sản phẩm yêu thích
# =============================================================================
class Favourite(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='favourites', verbose_name='Người dùng')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='favourites', verbose_name='Sản phẩm')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày thêm')

    class Meta:
        db_table = 'favourite'
        verbose_name = 'Sản phẩm yêu thích'
        verbose_name_plural = 'Sản phẩm yêu thích'
        unique_together = ('user', 'product')

    def __str__(self):
        return f'{self.user.phone} - {self.product.name}'


# =============================================================================
# OTP_Log - Lưu log gửi OTP (giới hạn số lượng/ngày)
# =============================================================================
class OTP_Log(models.Model):
    phone = models.CharField(max_length=15, verbose_name='Số điện thoại')
    otp = models.ForeignKey(OTP, on_delete=models.CASCADE, related_name='logs', verbose_name='OTP')
    ip_address = models.GenericIPAddressField(blank=True, null=True, verbose_name='Địa chỉ IP')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')

    class Meta:
        db_table = 'otp_log'
        verbose_name = 'Lịch sử OTP'
        verbose_name_plural = 'Lịch sử OTP'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.phone} - {self.created_at}'


# =============================================================================
# Setting - Cấu hình hệ thống
# =============================================================================
class Setting(models.Model):
    key = models.CharField(max_length=255, unique=True, verbose_name='Khóa')
    value = models.TextField(verbose_name='Giá trị')
    description = models.TextField(blank=True, null=True, verbose_name='Mô tả')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    class Meta:
        db_table = 'setting'
        verbose_name = 'Cấu hình'
        verbose_name_plural = 'Cấu hình'

    def __str__(self):
        return self.key


# =============================================================================
# Slide - Banner / Slide quảng cáo
# =============================================================================
class Slide(models.Model):
    title = models.CharField(max_length=255, verbose_name='Tiêu đề')
    subtitle = models.CharField(max_length=255, blank=True, null=True, verbose_name='Phụ đề')
    image = models.ImageField(upload_to='slides/', verbose_name='Ảnh')
    link = models.CharField(max_length=500, blank=True, null=True, verbose_name='Đường dẫn')
    sort_order = models.IntegerField(default=0, verbose_name='Thứ tự')
    is_active = models.BooleanField(default=True, verbose_name='Kích hoạt')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Ngày cập nhật')

    class Meta:
        db_table = 'slide'
        verbose_name = 'Slide/Banner'
        verbose_name_plural = 'Slide/Banner'
        ordering = ['sort_order']

    def __str__(self):
        return self.title
